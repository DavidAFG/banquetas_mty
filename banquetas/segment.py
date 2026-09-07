"""Stage 1b: semantic segmentation of the fetched images.

Uses a model pretrained on Mapillary Vistas, whose label set already contains
sidewalk, curb, curb cut, crosswalk and the street furniture that obstructs a
footway. Nothing is trained here and nothing is labelled by hand.

Label ids are resolved from the model config by substring rather than
hardcoded, and the resolved mapping is printed, because Vistas label names
differ between checkpoints and a silent mismatch would poison everything
downstream.

The label map for every image is written to disk as a single channel PNG.
That is the expensive artefact: measurement logic in stage 1c will change
several times, and it must not require rerunning the model each time.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CityConfig

DEFAULT_MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"

# Group name -> whole word phrases to match, and phrases to exclude.
# Names are normalised first: lowercased, punctuation to spaces. So both
# "construction--flat--sidewalk" and "Sidewalk" become "sidewalk", and the
# same table works across checkpoints with different naming conventions.
GROUPS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "sidewalk":   (("sidewalk",), ()),
    "ped_area":   (("pedestrian area",), ()),          # plazas, not footway
    "curb":       (("curb",), ("curb cut",)),
    "curb_cut":   (("curb cut",), ()),
    "crosswalk":  (("crosswalk",), ()),
    "lane_marking": (("lane marking",), ("crosswalk",)),
    "bike_lane":  (("bike lane",), ()),
    "road":       (("road", "service lane", "parking"), ()),
    "vegetation": (("vegetation",), ()),      # trees and shrubs
    "terrain":    (("terrain", "sand"), ()),  # grass and bare ground
    "pole":       (("pole", "street light", "traffic light"), ()),
    "sign":       (("traffic sign", "billboard", "banner", "signage"), ()),
    # Car Mount and Ego Vehicle are the camera rig, not traffic
    "car":        (("car", "truck", "bus", "van", "trailer", "other vehicle",
                    "motorcycle", "bicycle", "caravan", "wheeled slow"),
                   ("car mount", "ego vehicle")),
    "ego":        (("ego vehicle", "car mount"), ()),
    "person":     (("person", "bicyclist", "motorcyclist", "other rider"), ()),
    "furniture":  (("bench", "trash can", "mailbox", "fire hydrant", "phone booth",
                    "junction box", "cctv camera", "bike rack"), ()),
    "surface":    (("pothole", "manhole", "catch basin"), ()),   # pavement condition
    "building":   (("building",), ()),
    "fence":      (("fence", "guard rail", "barrier"), ()),
    "wall":       (("wall",), ()),
}

# groups that are the camera rig rather than the scene, masked before measuring
RIG_GROUPS = ("ego",)


def _norm(name: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", name.lower()).split())


def resolve_labels(id2label: dict[int, str], verbose: bool = True) -> dict[str, list[int]]:
    norm = {i: _norm(n) for i, n in id2label.items()}
    if verbose:
        print("  all classes in this checkpoint:")
        for i in sorted(id2label):
            print(f"    {i:3d}  {id2label[i]}")
    out: dict[str, list[int]] = {}
    for group, (wanted, excluded) in GROUPS.items():
        ids = []
        for i, n in norm.items():
            hit = any(re.search(rf"\b{re.escape(w)}\b", n) for w in wanted)
            bad = any(re.search(rf"\b{re.escape(x)}\b", n) for x in excluded)
            if hit and not bad:
                ids.append(i)
        out[group] = sorted(ids)
        if verbose:
            names = [id2label[i] for i in out[group]]
            print(f"    {group:11s} -> {', '.join(names) if names else '(NOTHING MATCHED)'}")
    if not out["sidewalk"]:
        raise RuntimeError("No sidewalk class in this checkpoint. Wrong model.")
    return out


def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run(city: CityConfig, model_id: str = DEFAULT_MODEL, max_size: int = 1024,
        limit: int | None = None, overlays: int = 0, refresh: bool = False,
        verbose: bool = True) -> pd.DataFrame:
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    meta = pd.read_csv(city.out_dir / "image_metadata.csv")
    meta = meta[meta["path"].notna()]
    if limit:
        meta = meta.head(limit)
    mask_dir = city.interim_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    ov_dir = city.out_dir / "overlays"
    if overlays:
        ov_dir.mkdir(parents=True, exist_ok=True)
    if refresh:
        for f in mask_dir.glob("*.png"):
            f.unlink()
        print("  cleared cached masks")

    dev = _device()
    if verbose:
        print(f"  device: {dev}")
        print(f"  model: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(dev).eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    if verbose:
        print(f"  {len(id2label)} classes in this checkpoint, resolving groups:")
    groups = resolve_labels(id2label, verbose=verbose)
    import json as _json
    (city.interim_dir / "label_map.json").write_text(_json.dumps(
        {"model": model_id, "id2label": id2label, "groups": groups}, indent=1))

    rows, done, skipped = [], 0, 0
    for r in meta.itertuples(index=False):
        img_id = str(r.id)
        mask_path = mask_dir / f"{img_id}.png"
        try:
            src = ImageOps.exif_transpose(Image.open(r.path))
        except Exception as exc:
            print(f"    unreadable {img_id}: {exc}")
            continue

        if mask_path.exists() and not refresh:
            seg = np.array(Image.open(mask_path))
            skipped += 1
        else:
            img = src.convert("RGB")
            if max(img.size) > max_size:
                scale = max_size / max(img.size)
                img = img.resize((int(img.width * scale), int(img.height * scale)))
            inputs = processor(images=img, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**inputs)
            seg = processor.post_process_semantic_segmentation(
                out, target_sizes=[img.size[::-1]])[0].cpu().numpy().astype(np.uint8)
            Image.fromarray(seg).save(mask_path, optimize=True)
            done += 1

        if overlays and len(rows) < overlays:
            _save_overlay(r.path, seg, groups, ov_dir / f"{img_id}.jpg")

        gray = np.array(src.convert("L").resize((seg.shape[1], seg.shape[0])))
        stat = _stats(seg, groups, img_id, getattr(r, "chunk_id", None), gray=gray)
        rows.append(stat)

        if verbose and (done + skipped) % 100 == 0:
            print(f"    {done + skipped}/{len(meta)}  (new {done}, cached {skipped})", flush=True)

    df = pd.DataFrame(rows)
    out_path = city.out_dir / "image_segstats.csv"
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"\n  segmented {done:,} images, reused {skipped:,} cached masks")
        print(f"  wrote {out_path}")
        if len(df):
            print("\n  sanity check, share of images where each group appears at all:")
            for g in GROUPS:
                col = f"{g}_share"
                if col in df:
                    print(f"    {g:11s} present in {100 * (df[col] > 0.001).mean():5.1f}% "
                          f"of images, median share when present "
                          f"{100 * df.loc[df[col] > 0.001, col].median():.2f}%")
    return df


OVERLAY_COLOURS = {
    "sidewalk": (0, 200, 255), "curb": (255, 140, 0), "curb_cut": (255, 0, 200),
    "crosswalk": (255, 255, 0), "road": (110, 110, 110), "vegetation": (0, 180, 60),
    "pole": (200, 0, 0), "sign": (255, 80, 80), "car": (140, 0, 220),
    "person": (255, 255, 255), "furniture": (0, 90, 255), "building": (90, 60, 40),
    "fence": (150, 130, 90),
}


def _save_overlay(img_path, seg, groups, dest):
    """Original beside a colourised mask, so failures can be seen not guessed."""
    import numpy as np
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    img = img.resize((seg.shape[1], seg.shape[0]))
    colour = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for group, ids in groups.items():
        if ids:
            colour[np.isin(seg, ids)] = OVERLAY_COLOURS.get(group, (255, 0, 255))
    blend = (np.asarray(img) * 0.45 + colour * 0.55).astype(np.uint8)
    pair = Image.new("RGB", (img.width * 2 + 8, img.height), (255, 255, 255))
    pair.paste(img, (0, 0))
    pair.paste(Image.fromarray(blend), (img.width + 8, 0))
    pair.save(dest, quality=88)


def ego_hood_mask(seg: np.ndarray, car_ids: list[int]) -> np.ndarray:
    """Mask the bonnet of the vehicle carrying the camera.

    These are dashcam frames shot through a windscreen, so the ego vehicle
    occupies the bottom of the image and the model labels it as a car. Left in,
    it would count as an obstruction in every single frame. It is identified
    as car pixels connected to the bottom edge, walking up column by column.
    """
    h, w = seg.shape
    hood = np.zeros_like(seg, dtype=bool)
    if not car_ids:
        return hood
    is_car = np.isin(seg, car_ids)
    for x in range(w):
        y = h - 1
        while y >= 0 and is_car[y, x]:
            hood[y, x] = True
            y -= 1
    return hood


def vignette_mask(gray: np.ndarray, thresh: int = 30) -> np.ndarray:
    """Mask the dark fisheye border.

    These lenses leave an unexposed rim, and the model has to call it
    something: here it comes out as car, right along the frame edges where
    the footway is measured. Identified as dark pixels connected to the left
    or right edge of each row, and to the top or bottom of each column.
    """
    h, w = gray.shape
    dark = gray < thresh
    mask = np.zeros_like(dark)
    for y in range(h):
        x = 0
        while x < w and dark[y, x]:
            mask[y, x] = True; x += 1
        x = w - 1
        while x >= 0 and dark[y, x]:
            mask[y, x] = True; x -= 1
    for x in range(w):
        y = 0
        while y < h and dark[y, x]:
            mask[y, x] = True; y += 1
        y = h - 1
        while y >= 0 and dark[y, x]:
            mask[y, x] = True; y -= 1
    return mask


def _stats(seg: np.ndarray, groups: dict[str, list[int]], img_id: str, chunk_id,
           gray: np.ndarray | None = None) -> dict:
    """Class shares over the whole frame and over each roadside separately.

    A forward facing camera looks down the carriageway. The footway, when
    there is one, appears as an oblique strip at the left or right edge of the
    lower half. Averaging over the bottom third measures the road surface and
    the bonnet, which is what the first pass was doing.

    Left and right are kept apart on purpose. INEGI records a sidewalk per
    frente de manzana, one per side of the street, so a per side measurement
    is the one that can be validated against it.
    """
    h, w = seg.shape
    ego_ids = groups.get("ego", [])
    hood = np.isin(seg, ego_ids) if ego_ids else np.zeros(seg.shape, dtype=bool)
    if not hood.any():
        # the class exists in the label set but this checkpoint never predicts
        # it on windscreen mounted footage, so fall back to the geometry
        hood = ego_hood_mask(seg, groups.get("car", []))
    vign = vignette_mask(gray) if gray is not None else np.zeros_like(hood)
    valid = ~hood & ~vign

    # regions are inset from the extreme edge, where fisheye distortion is
    # worst and the lens rim sits
    y0, y1 = int(h * 0.45), int(h * 0.97)
    regions = {
        "left": (slice(y0, y1), slice(int(w * 0.05), int(w * 0.32))),
        "right": (slice(y0, y1), slice(int(w * 0.68), int(w * 0.95))),
    }

    stat = {"id": img_id, "chunk_id": chunk_id, "h": h, "w": w,
            "hood_frac": float(hood.mean()), "vignette_frac": float(vign.mean())}
    for group, ids in groups.items():
        m = np.isin(seg, ids) & valid if ids else np.zeros_like(seg, dtype=bool)
        stat[f"{group}_share"] = float(m.sum()) / max(valid.sum(), 1)
        for side, (ys, xs) in regions.items():
            sub_valid = valid[ys, xs]
            stat[f"{group}_{side}"] = float((m[ys, xs]).sum()) / max(sub_valid.sum(), 1)
    return stat
