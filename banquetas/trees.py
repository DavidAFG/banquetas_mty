"""Side module: a street tree inventory accumulated from the same imagery pass.

This is deliberately NOT part of the audit's critical path. It rides along on
the segmentation stage: every time an image is processed for sidewalks, any
vegetation blob that sits at street level is also logged, projected to an
approximate ground position, and later clustered across images into candidate
tree locations.

Why it is worth having
  INEGI records ARBOLES as a binary per block face. A point inventory, even a
  rough one, is strictly more information, and street trees are the shade
  layer of the heat work.

Why it stays a side module
  Positions are approximate. A monocular camera with an assumed height gives
  distance errors that grow with distance, and a hedge is not a tree. Treat
  the output as candidate locations to be validated, never as a cadastre.

Interface expected from the segmentation stage (stage 2):
    detections = [
        {"bearing_offset_deg": float,   # relative to the camera's compass angle
         "distance_m": float,           # estimated ground distance
         "pixel_area": int,             # blob size, a rough confidence weight
         "class_name": "vegetation"},
        ...
    ]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

EARTH_R = 6371008.8


@dataclass
class TreeObservation:
    image_id: str
    lon: float
    lat: float
    captured_year: int | None
    distance_m: float
    pixel_area: int


def project_observation(lon: float, lat: float, compass_angle: float,
                        bearing_offset_deg: float, distance_m: float) -> tuple[float, float]:
    """Place a detection on the ground from the camera position.

    Simple great circle offset. At the distances involved, tens of metres,
    the spherical approximation is far more accurate than the distance
    estimate it consumes, so it is not the binding error.
    """
    bearing = math.radians((compass_angle + bearing_offset_deg) % 360.0)
    lat1, lon1 = math.radians(lat), math.radians(lon)
    d = distance_m / EARTH_R
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(d) * math.cos(lat1),
                             math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lon2), math.degrees(lat2)


def observations_from_image(image_row: pd.Series, detections: list[dict]) -> list[TreeObservation]:
    out = []
    compass = float(image_row.get("compass_angle") or 0.0)
    for det in detections:
        if det.get("class_name") not in ("vegetation", "tree"):
            continue
        lon, lat = project_observation(
            float(image_row["lon"]), float(image_row["lat"]), compass,
            float(det["bearing_offset_deg"]), float(det["distance_m"]))
        out.append(TreeObservation(
            image_id=str(image_row.get("id")),
            lon=lon, lat=lat,
            captured_year=int(image_row["year"]) if pd.notna(image_row.get("year")) else None,
            distance_m=float(det["distance_m"]),
            pixel_area=int(det.get("pixel_area", 0)),
        ))
    return out


def to_gdf(observations: list[TreeObservation]) -> gpd.GeoDataFrame:
    df = pd.DataFrame([asdict(o) for o in observations])
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs="EPSG:4326")


def cluster(obs: gpd.GeoDataFrame, crs, eps_m: float = 4.0, min_obs: int = 2) -> gpd.GeoDataFrame:
    """Collapse repeated sightings of the same tree into candidate locations.

    Dependency free single linkage clustering: buffer every observation by
    half of eps, dissolve, and treat each resulting blob as one tree.
    """
    if obs.empty:
        return obs.copy()
    pts = obs.to_crs(crs)
    blobs = pts.geometry.buffer(eps_m / 2.0).union_all()
    if blobs.geom_type == "Polygon":
        polys = [blobs]
    else:
        polys = list(blobs.geoms)
    clusters = gpd.GeoDataFrame({"cluster_id": range(len(polys))}, geometry=polys, crs=crs)

    joined = gpd.sjoin(pts, clusters, how="left", predicate="within")
    grouped = joined.groupby("cluster_id").agg(
        n_obs=("image_id", "nunique"),
        mean_pixel_area=("pixel_area", "mean"),
        first_year=("captured_year", "min"),
        last_year=("captured_year", "max"),
    ).reset_index()

    centroids = joined.dissolve(by="cluster_id").centroid
    out = grouped.merge(
        gpd.GeoDataFrame({"cluster_id": centroids.index, "geometry": centroids.values}, crs=crs),
        on="cluster_id")
    out = gpd.GeoDataFrame(out, crs=crs)
    out = out[out["n_obs"] >= min_obs]
    return out.to_crs("EPSG:4326")
