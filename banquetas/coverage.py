"""Stage 0: how much of the street network can actually be seen?

Coverage is measured on fixed length chunks rather than whole OSM segments,
because a two kilometre avenue with ten images clustered at one intersection
is not a covered avenue.

Outputs
  chunks.gpkg     one row per chunk, with image count and latest capture year
  segments.gpkg   one row per OSM segment, with coverage fraction
  coverage_by_tier.csv
  coverage_by_area.csv   when an area layer is supplied
"""
from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from .config import CityConfig

DEFAULT_CHUNK_M = 25.0
DEFAULT_SNAP_M = 12.0


def metric_crs(gdf: gpd.GeoDataFrame, city: CityConfig):
    if city.crs_metric:
        return city.crs_metric
    return gdf.estimate_utm_crs()


def cut_line(line: LineString, chunk_m: float) -> list[LineString]:
    """Split a line into pieces of roughly chunk_m metres.

    The line must already be in a metric CRS.
    """
    length = line.length
    if length <= chunk_m or length == 0:
        return [line]
    n = int(math.ceil(length / chunk_m))
    step = length / n
    pieces = []
    for i in range(n):
        a, b = i * step, (i + 1) * step
        p0 = line.interpolate(a)
        p1 = line.interpolate(b)
        # keep any original vertices that fall inside the piece so curves survive
        inner = [c for c in line.coords
                 if a < line.project(_pt(c)) < b]
        coords = [(p0.x, p0.y), *inner, (p1.x, p1.y)]
        # drop consecutive duplicates
        clean = [coords[0]]
        for c in coords[1:]:
            if c != clean[-1]:
                clean.append(c)
        if len(clean) >= 2:
            pieces.append(LineString(clean))
    return pieces or [line]


def _pt(coord):
    from shapely.geometry import Point
    return Point(coord[0], coord[1])


def chunk_network(edges: gpd.GeoDataFrame, crs, chunk_m: float = DEFAULT_CHUNK_M) -> gpd.GeoDataFrame:
    edges_m = edges.to_crs(crs)
    rows = []
    for row in edges_m.itertuples(index=False):
        for i, piece in enumerate(cut_line(row.geometry, chunk_m)):
            rows.append({
                "seg_id": row.seg_id,
                "chunk_idx": i,
                "tier": row.tier,
                "highway": row.highway,
                "name": getattr(row, "name", None),
                "maxspeed_kph": getattr(row, "maxspeed_kph", None),
                "osm_sidewalk_tag": getattr(row, "sidewalk", None),
                "length_m": piece.length,
                "geometry": piece,
            })
    chunks = gpd.GeoDataFrame(rows, crs=crs)
    chunks["chunk_id"] = chunks.index.astype(int)
    return chunks


def attach_images(chunks: gpd.GeoDataFrame, points: gpd.GeoDataFrame,
                  snap_m: float = DEFAULT_SNAP_M) -> gpd.GeoDataFrame:
    """Assign each image point to its nearest chunk within snap_m and count.

    Nearest assignment rather than a buffer join, so an image at an
    intersection is counted once instead of once per touching segment.
    """
    pts = points.to_crs(chunks.crs)
    joined = gpd.sjoin_nearest(
        pts, chunks[["chunk_id", "geometry"]],
        how="inner", max_distance=snap_m, distance_col="snap_dist_m")
    # sjoin_nearest can return ties; keep one row per image point
    joined = joined[~joined.index.duplicated(keep="first")]
    if joined.empty:
        chunks["n_images"] = 0
        chunks["latest_year"] = pd.NA
        chunks["n_pano"] = 0
        return chunks

    agg = {"n_images": ("chunk_id", "size")}
    if "year" in joined.columns:
        agg["latest_year"] = ("year", "max")
        agg["earliest_year"] = ("year", "min")
    if "is_pano" in joined.columns:
        joined["_pano"] = joined["is_pano"].astype("boolean").fillna(False).astype(int)
        agg["n_pano"] = ("_pano", "sum")

    stats = joined.groupby("chunk_id").agg(**agg).reset_index()
    out = chunks.merge(stats, on="chunk_id", how="left")
    out["n_images"] = out["n_images"].fillna(0).astype(int)
    if "n_pano" in out.columns:
        out["n_pano"] = out["n_pano"].fillna(0).astype(int)
    out["observed"] = out["n_images"] > 0
    return out


def summarise_segments(chunks: gpd.GeoDataFrame) -> pd.DataFrame:
    g = chunks.groupby("seg_id")
    seg = g.agg(
        tier=("tier", "first"),
        highway=("highway", "first"),
        name=("name", "first"),
        maxspeed_kph=("maxspeed_kph", "first"),
        osm_sidewalk_tag=("osm_sidewalk_tag", "first"),
        length_m=("length_m", "sum"),
        n_chunks=("chunk_id", "size"),
        n_chunks_observed=("observed", "sum"),
        n_images=("n_images", "sum"),
    ).reset_index()
    seg["coverage_frac"] = seg["n_chunks_observed"] / seg["n_chunks"]
    if "latest_year" in chunks.columns:
        seg = seg.merge(g["latest_year"].max().rename("latest_year").reset_index(), on="seg_id")
    return seg


def summarise_by(chunks: gpd.GeoDataFrame, by: str) -> pd.DataFrame:
    """Length weighted coverage by any grouping column."""
    df = chunks.copy()
    df["obs_len"] = df["length_m"] * df["observed"].astype(int)
    out = df.groupby(by, dropna=False).agg(
        km_total=("length_m", lambda s: s.sum() / 1000.0),
        km_observed=("obs_len", lambda s: s.sum() / 1000.0),
        n_chunks=("chunk_id", "size"),
        n_images=("n_images", "sum"),
    ).reset_index()
    out["coverage_pct"] = 100.0 * out["km_observed"] / out["km_total"].replace(0, np.nan)
    return out.sort_values("km_total", ascending=False)


def tag_areas(chunks: gpd.GeoDataFrame, areas: gpd.GeoDataFrame, id_col: str) -> gpd.GeoDataFrame:
    """Attach an area identifier (municipio, AGEB, colonia) to each chunk.

    The chunk is assigned to the area containing its midpoint, which avoids
    double counting chunks that straddle a boundary.
    """
    mids = chunks.copy()
    mids["geometry"] = mids.geometry.interpolate(0.5, normalized=True)
    joined = gpd.sjoin(mids[["chunk_id", "geometry"]], areas.to_crs(chunks.crs)[[id_col, "geometry"]],
                       how="left", predicate="within")
    joined = joined.drop_duplicates(subset="chunk_id")[["chunk_id", id_col]]
    return chunks.merge(joined, on="chunk_id", how="left")


def run(city: CityConfig, edges: gpd.GeoDataFrame, points: gpd.GeoDataFrame,
        areas: gpd.GeoDataFrame | None = None, area_id_col: str | None = None,
        chunk_m: float = DEFAULT_CHUNK_M, snap_m: float = DEFAULT_SNAP_M,
        verbose: bool = True) -> dict:
    crs = metric_crs(edges, city)
    if verbose:
        print(f"  metric CRS: {crs}")
    chunks = chunk_network(edges, crs, chunk_m=chunk_m)
    if verbose:
        print(f"  {len(chunks)} chunks of ~{chunk_m:.0f} m from {len(edges)} segments")
    chunks = attach_images(chunks, points, snap_m=snap_m)

    if areas is not None and area_id_col:
        chunks = tag_areas(chunks, areas, area_id_col)

    segments = summarise_segments(chunks)
    by_tier = summarise_by(chunks, "tier")

    chunks.to_file(city.out_dir / "chunks.gpkg", driver="GPKG", layer="chunks")
    seg_geom = edges[["seg_id", "geometry"]].merge(segments, on="seg_id")
    gpd.GeoDataFrame(seg_geom, crs=edges.crs).to_file(
        city.out_dir / "segments.gpkg", driver="GPKG", layer="segments")
    by_tier.to_csv(city.out_dir / "coverage_by_tier.csv", index=False)

    result = {"chunks": chunks, "segments": segments, "by_tier": by_tier}
    if areas is not None and area_id_col:
        by_area = summarise_by(chunks, area_id_col)
        by_area.to_csv(city.out_dir / "coverage_by_area.csv", index=False)
        result["by_area"] = by_area

    if verbose:
        print("\nCoverage by road tier")
        print(by_tier.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    return result
