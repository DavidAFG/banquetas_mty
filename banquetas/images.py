"""Stage 1a, part two: pick a few images per chunk and fetch them.

Selection matters more than volume. A median observed chunk has seven images
and many have dozens, but they are mostly the same vehicle passing once, so
the marginal frame adds nothing. What adds information is looking the other
way, because a forward facing camera sees one side of the street properly.

So images are chosen by: not panoramic, highest quality_score, and spread
across opposing compass bearings so both sides of the chunk get a look.

Camera intrinsics are fetched with the metadata, because the width step
needs them and they are not available later without another round trip.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from .config import CityConfig, mapillary_token

GRAPH = "https://graph.mapillary.com/images"
FIELDS = ("id,thumb_1024_url,camera_parameters,camera_type,compass_angle,"
          "computed_compass_angle,computed_geometry,captured_at,height,width,"
          "quality_score,is_pano,sequence")
BATCH = 50


def select(city: CityConfig, sample: gpd.GeoDataFrame, k: int = 3,
           snap_m: float = 12.0, verbose: bool = True) -> pd.DataFrame:
    """Assign image points to sampled chunks and keep the best k per chunk."""
    pts = gpd.read_file(city.interim_dir / "mapillary_points.gpkg", layer="images")
    pts = pts.to_crs(sample.crs)
    if verbose:
        print(f"  image points loaded: {len(pts):,}")

    j = gpd.sjoin_nearest(pts, sample[["chunk_id", "geometry"]], how="inner",
                          max_distance=snap_m, distance_col="snap_dist_m")
    j = j[~j.index.duplicated(keep="first")]
    if verbose:
        print(f"  image points on sampled chunks: {len(j):,}")

    if "is_pano" in j.columns:
        j = j[~j["is_pano"].astype("boolean").fillna(False)]
    j["quality_score"] = (pd.to_numeric(j["quality_score"], errors="coerce").fillna(0)
                          if "quality_score" in j.columns else 0.0)
    ang = (pd.to_numeric(j["compass_angle"], errors="coerce")
           if "compass_angle" in j.columns else pd.Series(0.0, index=j.index))
    j["angle_bin"] = (ang.fillna(0).mod(360) // 180).astype(int)

    # rank within bearing bin, then interleave: the first pick from every bin
    # comes before the second pick from any bin, so both sides get seen first
    j["_r"] = j.groupby(["chunk_id", "angle_bin"])["quality_score"].rank(
        method="first", ascending=False)
    j = j.sort_values(["chunk_id", "_r", "quality_score"], ascending=[True, True, False])
    picked = j.groupby("chunk_id", as_index=False).head(k)

    cols = [c for c in ("id", "chunk_id", "angle_bin", "quality_score",
                        "compass_angle", "captured_at", "year", "snap_dist_m",
                        "sequence_id", "lon", "lat") if c in picked.columns]
    out = pd.DataFrame(picked[cols])
    if verbose:
        per = out.groupby("chunk_id").size()
        print(f"  selected {len(out):,} images for {per.size:,} chunks "
              f"(mean {per.mean():.1f} per chunk, both bearings on "
              f"{100 * (out.groupby('chunk_id')['angle_bin'].nunique() > 1).mean():.0f}%)")
    return out


def fetch_metadata(image_ids: list[str], token: str, verbose: bool = True) -> pd.DataFrame:
    """Graph API metadata, including camera intrinsics, in batches."""
    session = requests.Session()
    rows, delay = [], 1.0
    for i in range(0, len(image_ids), BATCH):
        batch = image_ids[i:i + BATCH]
        for attempt in range(5):
            r = session.get(GRAPH, params={"access_token": token, "fields": FIELDS,
                                           "image_ids": ",".join(map(str, batch))}, timeout=60)
            if r.status_code == 200:
                rows.extend(r.json().get("data", []))
                break
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay); delay = min(delay * 2, 60); continue
            raise RuntimeError(f"Graph API {r.status_code}: {r.text[:200]}")
        if verbose and (i // BATCH) % 10 == 0:
            print(f"    metadata {min(i + BATCH, len(image_ids))}/{len(image_ids)}", flush=True)
    df = pd.DataFrame(rows)
    for col in ("camera_parameters", "computed_geometry"):
        if col in df.columns:
            df[col] = df[col].map(lambda v: json.dumps(v) if v is not None else None)
    return df


def download(meta: pd.DataFrame, out_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """Fetch thumbnails, skipping any already on disk so runs resume."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    paths, ok = [], 0
    for n, row in enumerate(meta.itertuples(index=False), 1):
        dest = out_dir / f"{row.id}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            paths.append(str(dest)); ok += 1; continue
        url = getattr(row, "thumb_1024_url", None)
        if not url:
            paths.append(None); continue
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            paths.append(str(dest)); ok += 1
        except Exception as exc:
            print(f"    failed {row.id}: {exc}")
            paths.append(None)
        if verbose and n % 250 == 0:
            print(f"    images {n}/{len(meta)}", flush=True)
    meta = meta.copy()
    meta["path"] = paths
    if verbose:
        mb = sum(Path(p).stat().st_size for p in paths if p) / 1e6
        print(f"  downloaded {ok:,} of {len(meta):,} images, {mb:,.0f} MB on disk")
    return meta


def run(city: CityConfig, k: int = 3, snap_m: float = 12.0, dry_run: bool = False,
        verbose: bool = True) -> pd.DataFrame:
    sample = gpd.read_file(city.out_dir / "sample_chunks.gpkg", layer="sample")
    picked = select(city, sample, k=k, snap_m=snap_m, verbose=verbose)
    picked.to_csv(city.out_dir / "sample_images.csv", index=False)

    if dry_run:
        print(f"\n  dry run: would fetch {len(picked):,} images, roughly "
              f"{len(picked) * 0.15:,.0f} MB at 1024 px")
        return picked

    token = mapillary_token()
    meta = fetch_metadata(picked["id"].astype(str).tolist(), token, verbose=verbose)
    meta = meta.merge(picked[["id", "chunk_id"]].astype({"id": str}),
                      on="id", how="left")
    meta = download(meta, city.raw_dir / "images", verbose=verbose)
    meta.to_csv(city.out_dir / "image_metadata.csv", index=False)
    if verbose:
        print(f"  wrote {city.out_dir / 'image_metadata.csv'}")
        if "camera_type" in meta.columns:
            print("  camera types: " + ", ".join(
                f"{k}={v}" for k, v in meta["camera_type"].value_counts().items()))
    return meta
