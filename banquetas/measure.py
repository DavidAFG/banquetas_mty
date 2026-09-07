"""Stage 1c: per image measurement, aggregated to chunk and block face.

Reads the cached label maps rather than the model, so the measurement
definition can change as often as it needs to without another GPU pass.

Block faces: the near kerb is on the vehicle's right, so the face a frame
sees lies at compass_angle + 90. Two frames driven in opposite directions on
the same chunk therefore see the two different sides of the street, which is
the unit INEGI records BANQUETA on.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import CityConfig
from .geometry import measure_image, parse_camera

# a sidewalk narrower than this is not usable by two people or a wheelchair
NARROW_M = 1.2
# Sidewalk pixels on the ground needed to call a frame present. Tuned against
# 177 hand labelled frames: accuracy is flat between 50 and 200 and falls away
# outside, so this is a plateau rather than a knife edge.
PRESENCE_PX = 100


def load_groups(city: CityConfig) -> dict[str, list[int]]:
    path = city.interim_dir / "label_map.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Rerun `segment` once, it now writes the label mapping.")
    return {k: list(map(int, v)) for k, v in json.loads(path.read_text())["groups"].items()}


def face_id(compass: float) -> float:
    """Bearing of the block face the camera's near side is looking at."""
    return (float(compass) + 90.0) % 360.0


def _same_face(a: float, b: float) -> bool:
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d < 90.0


def run(city: CityConfig, limit: int | None = None, verbose: bool = True) -> pd.DataFrame:
    from PIL import Image

    groups = load_groups(city)
    meta = pd.read_csv(city.out_dir / "image_metadata.csv")
    meta = meta[meta["path"].notna()]
    if limit:
        meta = meta.head(limit)

    h_m, pitch = city.camera_height_m, city.camera_pitch_deg
    if verbose:
        print(f"  camera height {h_m} m, pitch {pitch} deg  "
              f"(set in config/cities/{city.name}.yml)")
        print(f"  measuring {len(meta):,} images")

    mask_dir = city.interim_dir / "masks"
    rows = []
    for n, r in enumerate(meta.itertuples(index=False), 1):
        mp = mask_dir / f"{r.id}.png"
        if not mp.exists():
            continue
        seg = np.array(Image.open(mp))
        res = measure_image(seg, groups, parse_camera(r), h_m, pitch, "right")
        res["id"] = str(r.id)
        res["chunk_id"] = getattr(r, "chunk_id", None)
        compass = getattr(r, "computed_compass_angle", None) or getattr(r, "compass_angle", 0.0)
        res["face_bearing"] = face_id(compass if pd.notna(compass) else 0.0)
        rows.append(res)
        if verbose and n % 500 == 0:
            print(f"    {n}/{len(meta)}", flush=True)

    img = pd.DataFrame(rows)
    img["present"] = img["n_sidewalk_px"] >= PRESENCE_PX
    img.to_csv(city.out_dir / "image_measurements.csv", index=False)

    faces = []
    for cid, g in img.groupby("chunk_id"):
        anchor = g["face_bearing"].iloc[0]
        g = g.assign(face=["A" if _same_face(b, anchor) else "B" for b in g["face_bearing"]])
        for face, gg in g.groupby("face"):
            faces.append({
                "chunk_id": cid, "face": face, "n_images": len(gg),
                "face_bearing": float(gg["face_bearing"].median()),
                # a footway seen in ANY pass over the face is a footway. Requiring
                # a majority of frames costs 11 points of recall for nothing:
                # validation gives 0.854 against 0.740 at the same precision.
                "present_any": bool(gg["present"].any()),
                "present_rate": float(gg["present"].mean()),
                "width_m": float(np.nanmedian(gg["width_m"])) if gg["width_m"].notna().any() else np.nan,
                "kerb_offset_m": float(np.nanmedian(gg["kerb_offset_m"])) if gg["kerb_offset_m"].notna().any() else np.nan,
                "road_width_m": float(np.nanmedian(gg["road_width_m"])) if gg["road_width_m"].notna().any() else np.nan,
                "obstructed_frac": float(gg["obstructed_frac"].mean()),
                "canopy_frac": float(gg["canopy_frac"].mean()),
                "rooted_frac": float(gg["rooted_frac"].mean()),
            })
    face_df = pd.DataFrame(faces)
    face_df["class"] = np.where(
        ~face_df["present_any"], "absent",
        np.where(face_df["width_m"] < NARROW_M, "narrow", "adequate"))
    face_df.to_csv(city.out_dir / "blockface_measurements.csv", index=False)

    if verbose:
        print(f"\n  {len(img):,} images measured, {len(face_df):,} block faces")
        print(f"  sidewalk present in {100 * img['present'].mean():.1f}% of images")
        w = img["width_m"].dropna()
        if len(w):
            print(f"  width, images with a usable estimate: {len(w):,} "
                  f"({100 * len(w) / len(img):.0f}%)")
            print("    " + "  ".join(f"p{p}={np.percentile(w, p):.2f}" for p in (10, 25, 50, 75, 90)))
        rw = img["road_width_m"].dropna()
        if len(rw):
            print(f"\n  CALIBRATION CHECK, carriageway width: median {np.median(rw):.2f} m, "
                  f"p25 {np.percentile(rw, 25):.2f}, p75 {np.percentile(rw, 75):.2f}")
            print("    A two lane street should land near 7 m and a four lane near 14 m.")
            print(f"    If these are systematically off, the camera height in "
                  f"config/cities/{city.name}.yml is wrong, and every width scales with it.")
        lw = img["lane_width_m"].dropna()
        if len(lw):
            med = float(np.median(lw))
            implied = h_m * 3.3 / med
            q1, q3 = np.percentile(lw, [25, 75])
            print(f"\n  LANE SPACING CHECK, {len(lw):,} images with lane markings: "
                  f"median {med:.2f} m  (p25 {q1:.2f}, p75 {q3:.2f})")
            print(f"    A marked lane is 3.2 to 3.5 m. At the assumed height of {h_m} m "
                  f"this reads {med:.2f} m,")
            print(f"    which implies a true camera height of about {implied:.2f} m "
                  f"if lanes are 3.3 m.")
            print(f"    Scale every width by {3.3 / med:.3f} to correct, or set "
                  f"height_m: {implied:.2f} in the city config and rerun.")

        if "canopy_frac" in img.columns:
            withwalk = img[img["present"]]
            print(f"\n  greenery, among {len(withwalk):,} frames with a footway")
            print(f"    canopy overhead: median {withwalk['canopy_frac'].median():.2f} "
                  f"of the footway, none at all on "
                  f"{100 * (withwalk['canopy_frac'] < 0.01).mean():.0f}% of frames")
            print(f"    something rooted beside it on "
                  f"{100 * (withwalk['rooted_frac'] > 0.01).mean():.0f}% of frames")
            print("    NOT yet validated. INEGI's ARBOLES variable is the check,")
            print("    recorded per frente de manzana, the same unit as this.")

        print("\n  block face classes")
        print(face_df["class"].value_counts().to_string())
    return face_df
