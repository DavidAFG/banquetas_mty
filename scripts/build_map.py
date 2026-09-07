"""Build the interactive audit map from the pilot measurements.

Two design decisions worth stating, because both are contestable.

Colour is an ordinal single hue ramp, not red/amber/green. A three step
red/amber/green scale fails colour vision deficiency badly: green against red
measures a deuteranopia delta E of 4.1, which is indistinguishable. The blue
ordinal ramp passes every check, reads correctly in greyscale, and is paired
with line width as a redundant channel so the encoding never rests on hue
alone. The worst class always carries the most contrast against the surface,
so it is darkest on light and lightest on dark.

Every face links to the frame it was measured from, with the segmentation mask
drawn over it. That is what makes this an audit rather than a model output:
any claim on the map can be checked by eye in two seconds.
"""
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banquetas.config import load_city
from banquetas.segment import OVERLAY_COLOURS

NON_PEDESTRIAN = ("motorway", "motorway_link", "trunk", "trunk_link")
OFFSET_M = 4.0
OVERLAY_W = 640


def face_geometries(faces, chunks):
    """Shift each chunk line onto the side it describes, so the two faces of a
    street draw as two parallel lines rather than one on top of the other.

    Returns the filtered face table and its geometries together, so the two can
    never fall out of alignment.
    """
    from shapely.affinity import translate

    geo = chunks.drop_duplicates("chunk_id").set_index("chunk_id")["geometry"]
    keep = faces[faces["chunk_id"].isin(geo.index)].reset_index(drop=True)
    geoms = []
    for cid, bearing in zip(keep["chunk_id"], keep["face_bearing"]):
        br = math.radians(float(bearing))
        geoms.append(translate(geo.loc[cid],
                               xoff=math.sin(br) * OFFSET_M,
                               yoff=math.cos(br) * OFFSET_M))
    return keep, geoms


def build_overlays(city, wanted, verbose=True):
    """One composited frame per block face, small enough to keep 3,000 of them."""
    from PIL import Image, ImageOps
    groups = json.loads((city.interim_dir / "label_map.json").read_text())["groups"]
    groups = {k: list(map(int, v)) for k, v in groups.items()}
    mask_dir = city.interim_dir / "masks"
    img_dir = city.raw_dir / "images"
    out_dir = city.out_dir / "map" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    made = skipped = 0
    for n, img_id in enumerate(wanted, 1):
        dest = out_dir / f"{img_id}.jpg"
        if dest.exists():
            skipped += 1
            continue
        mp, ip = mask_dir / f"{img_id}.png", img_dir / f"{img_id}.jpg"
        if not (mp.exists() and ip.exists()):
            continue
        seg = np.array(Image.open(mp))
        img = ImageOps.exif_transpose(Image.open(ip)).convert("RGB").resize(
            (seg.shape[1], seg.shape[0]))
        colour = np.zeros((*seg.shape, 3), dtype=np.uint8)
        for g, ids in groups.items():
            if ids and g in OVERLAY_COLOURS:
                colour[np.isin(seg, ids)] = OVERLAY_COLOURS[g]
        blend = (np.asarray(img) * 0.5 + colour * 0.5).astype(np.uint8)
        pair = Image.new("RGB", (img.width, img.height * 2 + 6), (255, 255, 255))
        pair.paste(img, (0, 0))
        pair.paste(Image.fromarray(blend), (0, img.height + 6))
        w = OVERLAY_W
        pair = pair.resize((w, int(pair.height * w / pair.width)))
        pair.save(dest, quality=72, optimize=True)
        made += 1
        if verbose and n % 400 == 0:
            print(f"    evidence {n}/{len(wanted)}", flush=True)
    if verbose:
        print(f"  evidence frames: {made} written, {skipped} already present")


def main(city_name="monterrey", overlays="yes"):
    city = load_city(city_name)
    faces = pd.read_csv(city.out_dir / "blockface_measurements.csv")
    chunks = gpd.read_file(city.out_dir / "sample_chunks.gpkg", layer="sample")
    imgs = pd.read_csv(city.out_dir / "image_measurements.csv")

    have = [c for c in ("chunk_id", "mun_name", "ses_decile", "tier", "length_m")
            if c in chunks.columns]
    ctx = pd.DataFrame(chunks.drop(columns="geometry"))[have]
    faces = faces.merge(ctx, on="chunk_id", how="left")

    # the road tag lives in chunks.gpkg; read it straight out rather than
    # depending on whichever intermediate happened to carry it forward
    import sqlite3
    with sqlite3.connect(city.out_dir / "chunks.gpkg") as con:
        hw = pd.read_sql("SELECT chunk_id, highway FROM chunks", con)
    faces = faces.merge(hw, on="chunk_id", how="left")

    # Pedestrians are prohibited on motorways and trunk roads, so a missing
    # footway there is not a finding. Shown on the map, excluded from the
    # statistics, rather than quietly deleted.
    faces.loc[faces["highway"].isin(NON_PEDESTRIAN), "class"] = "n/a"
    n_na = int((faces["class"] == "n/a").sum())
    print(f"  {n_na:,} faces on motorway or trunk marked n/a "
          f"({100 * n_na / max(len(faces), 1):.1f}%)")

    # one representative frame per face: the one with the most sidewalk evidence,
    # or failing that the first, so absent faces are inspectable too
    def same(a, b):
        return abs((a - b + 180) % 360 - 180) < 90
    pick = {}
    for cid, g in imgs.groupby("chunk_id"):
        anchor = g["face_bearing"].iloc[0]
        for r in g.itertuples(index=False):
            f = "A" if same(r.face_bearing, anchor) else "B"
            key = (cid, f)
            px = float(r.n_sidewalk_px) if pd.notna(r.n_sidewalk_px) else 0.0
            cur = pick.get(key)
            if cur is None or px > cur[1]:
                pick[key] = (str(r.id), px)
    faces["img"] = [pick.get((c, f), ("", 0))[0]
                    for c, f in zip(faces["chunk_id"], faces["face"])]

    # what the other side of the same street says, so a red line beside a blue
    # one can be judged rather than guessed at
    sib = {}
    for cid, g in faces.groupby("chunk_id"):
        if len(g) == 2:
            a, bb = g.iloc[0], g.iloc[1]
            sib[(cid, a["face"])] = bb["class"]
            sib[(cid, bb["face"])] = a["class"]
    faces["other_side"] = [sib.get((c, f), "not measured")
                           for c, f in zip(faces["chunk_id"], faces["face"])]
    faces["disagrees"] = [
        (o in ("absent", "narrow", "adequate")) and (o != c)
        for c, o in zip(faces["class"], faces["other_side"])]

    faces, geoms = face_geometries(faces, chunks)
    gdf = gpd.GeoDataFrame(faces, geometry=geoms, crs=chunks.crs).to_crs("EPSG:4326")

    for c in ("width_m", "kerb_offset_m", "road_width_m", "obstructed_frac",
              "present_rate", "canopy_frac", "rooted_frac"):
        gdf[c] = gdf[c].round(3)
    gdf["ses_decile"] = gdf["ses_decile"].astype(str)

    out_dir = city.out_dir / "map"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["chunk_id", "face", "class", "width_m", "present_rate", "obstructed_frac",
            "n_images", "road_width_m", "kerb_offset_m", "mun_name", "ses_decile",
            "tier", "highway", "length_m", "img", "other_side", "disagrees",
            "canopy_frac", "rooted_frac"]
    gj = json.loads(gdf[cols + ["geometry"]].to_json())
    (out_dir / "data.geojson").write_text(json.dumps(gj, separators=(",", ":")))
    print(f"  {len(gdf):,} block faces written to data.geojson "
          f"({(out_dir / 'data.geojson').stat().st_size / 1e6:.1f} MB)")

    if str(overlays).lower() in ("yes", "true", "1"):
        build_overlays(city, [i for i in gdf["img"].tolist() if i])

    tpl = Path(__file__).resolve().parents[1] / "banquetas" / "map" / "template.html"
    payload = json.dumps(gj, separators=(",", ":")).replace("<", "\\u003c")
    html = (tpl.read_text()
            .replace("__CITY__", city.label)
            .replace("__DATA__", payload))
    (out_dir / "index.html").write_text(html)
    print(f"  index.html is {(out_dir / 'index.html').stat().st_size / 1e6:.1f} MB "
          f"with the data inlined")
    print(f"  open {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
