"""Harvest Mapillary image point metadata from vector tiles.

Stage 0 never downloads a single photograph. It only asks where images
exist, which is enough to measure coverage and costs almost nothing.

Tiles are cached on disk, so a re-run after an interruption resumes instead
of refetching. The Mapillary rate limit is generous (50k requests per day)
but this module still backs off politely.
"""
from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import mercantile
import pandas as pd
import requests
from shapely.geometry import Point
from vt2geojson.tools import vt_bytes_to_geojson

from .config import CityConfig, mapillary_token

TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"
IMAGE_LAYER = "image"


def tiles_for_bbox(bbox, zoom: int) -> list[mercantile.Tile]:
    west, south, east, north = bbox
    return list(mercantile.tiles(west, south, east, north, zooms=[zoom]))


def _tile_path(cache: Path, t: mercantile.Tile) -> Path:
    p = cache / str(t.z) / str(t.x)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{t.y}.mvt"


def fetch_tile(t: mercantile.Tile, token: str, cache: Path,
               session: requests.Session | None = None,
               max_retries: int = 5) -> bytes:
    """Return raw tile bytes, from cache when available.

    An empty file in the cache means the tile genuinely holds no imagery.
    """
    path = _tile_path(cache, t)
    if path.exists():
        return path.read_bytes()

    session = session or requests.Session()
    url = TILE_URL.format(z=t.z, x=t.x, y=t.y)
    delay = 1.0
    for attempt in range(max_retries):
        r = session.get(url, params={"access_token": token}, timeout=60)
        if r.status_code == 200:
            path.write_bytes(r.content)
            return r.content
        if r.status_code == 404:
            path.write_bytes(b"")
            return b""
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        raise RuntimeError(f"Mapillary tile {t.z}/{t.x}/{t.y} failed: {r.status_code} {r.text[:200]}")
    raise RuntimeError(f"Mapillary tile {t.z}/{t.x}/{t.y} still failing after {max_retries} retries")


def tile_to_records(raw: bytes, t: mercantile.Tile) -> list[dict]:
    if not raw:
        return []
    fc = vt_bytes_to_geojson(raw, t.x, t.y, t.z, layer=IMAGE_LAYER)
    records = []
    for feat in fc.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        lon, lat = geom["coordinates"][:2]
        props = dict(feat.get("properties") or {})
        props["lon"] = lon
        props["lat"] = lat
        records.append(props)
    return records


def harvest(city: CityConfig, verbose: bool = True) -> gpd.GeoDataFrame:
    """Fetch every image point in the city bbox and return it as a GeoDataFrame."""
    token = mapillary_token()
    zoom = int(city.mapillary.get("zoom", 14))
    tiles = tiles_for_bbox(city.bbox, zoom)
    session = requests.Session()

    rows: list[dict] = []
    for i, t in enumerate(tiles, 1):
        raw = fetch_tile(t, token, city.tile_cache, session=session)
        rows.extend(tile_to_records(raw, t))
        if verbose and (i % 25 == 0 or i == len(tiles)):
            print(f"  tiles {i}/{len(tiles)}  points so far: {len(rows)}", flush=True)

    if not rows:
        raise RuntimeError("No image points returned. Check the bbox and the token.")

    df = pd.DataFrame(rows)
    if "captured_at" in df.columns:
        # Mapillary reports capture time in milliseconds since epoch.
        df["captured_dt"] = pd.to_datetime(df["captured_at"], unit="ms", errors="coerce")
        df["year"] = df["captured_dt"].dt.year
    df = df.drop_duplicates(subset=[c for c in ("id",) if c in df.columns] or None)

    min_year = city.mapillary.get("min_year")
    if min_year and "year" in df.columns:
        before = len(df)
        df = df[df["year"] >= int(min_year)]
        if verbose:
            print(f"  year filter >= {min_year}: kept {len(df)} of {before} points")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )
    out = city.interim_dir / "mapillary_points.gpkg"
    gdf.to_file(out, driver="GPKG", layer="images")
    if verbose:
        print(f"  wrote {len(gdf)} image points to {out}")
    return gdf
