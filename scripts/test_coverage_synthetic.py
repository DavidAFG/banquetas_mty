"""Offline sanity check for the stage 0 geometry logic.

Builds a small synthetic grid city with known coverage, runs the real
coverage code over it, and asserts the numbers come out as expected. This
runs with no network and no Mapillary token.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from banquetas import coverage as cov
from banquetas.config import CityConfig

LON0, LAT0 = -100.30, 25.68
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.radians(LAT0))


def to_lonlat(x_m, y_m):
    return LON0 + x_m / M_PER_DEG_LON, LAT0 + y_m / M_PER_DEG_LAT


def build_city():
    return CityConfig(
        name="_synthetic", label="Synthetic grid", country="MX",
        bbox=(LON0 - 0.02, LAT0 - 0.02, LON0 + 0.02, LAT0 + 0.02),
        mapillary={"zoom": 14}, osm={}, crs_metric="EPSG:32614",
        tiers={"arterial": ["primary"], "local": ["residential"]},
    )


def build_edges():
    """Three streets of 1000 m each: one arterial, two residential."""
    rows = []
    specs = [
        ("primary", 0),      # y = 0
        ("residential", 200),
        ("residential", 400),
    ]
    for i, (highway, y) in enumerate(specs):
        coords = [to_lonlat(x, y) for x in (0, 250, 500, 750, 1000)]
        rows.append({"seg_id": i, "highway": highway, "name": f"street_{i}",
                     "maxspeed_kph": 50.0, "sidewalk": None,
                     "tier": "arterial" if highway == "primary" else "local",
                     "geometry": LineString(coords)})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def build_points():
    """Full coverage of the arterial, half of street 1, nothing on street 2."""
    pts = []
    for x in range(0, 1001, 10):          # arterial, every 10 m
        pts.append((*to_lonlat(x, 0), 2023))
    for x in range(0, 501, 10):           # street 1, first half only
        pts.append((*to_lonlat(x, 200), 2021))
    # a decoy 60 m away from every street, must not be snapped to anything
    pts.append((*to_lonlat(500, 700), 2022))
    df = pd.DataFrame(pts, columns=["lon", "lat", "year"])
    df["id"] = [f"img{i}" for i in range(len(df))]
    df["compass_angle"] = 90.0
    df["is_pano"] = False
    return gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs="EPSG:4326")


def main():
    city = build_city()
    edges, points = build_edges(), build_points()

    res = cov.run(city, edges, points, chunk_m=25.0, snap_m=12.0, verbose=False)
    seg = res["segments"].set_index("seg_id")
    tier = res["by_tier"].set_index("tier")

    print(seg[["tier", "length_m", "n_chunks", "n_chunks_observed", "coverage_frac", "n_images"]]
          .to_string(float_format=lambda v: f"{v:,.2f}"))
    print()
    print(tier[["km_total", "km_observed", "coverage_pct"]].to_string(float_format=lambda v: f"{v:,.2f}"))

    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        ok = ok and cond

    print("\nchecks")
    check("segment lengths near 1000 m", all(abs(v - 1000) < 5 for v in seg["length_m"]))
    check("about 40 chunks per 1000 m segment", set(seg["n_chunks"]) <= {40, 41})
    check("arterial fully covered", seg.loc[0, "coverage_frac"] == 1.0)
    check("street 1 about half covered", 0.45 <= seg.loc[1, "coverage_frac"] <= 0.55)
    check("street 2 not covered", seg.loc[2, "coverage_frac"] == 0.0)
    check("decoy point snapped to nothing", seg["n_images"].sum() == len(points) - 1)
    check("arterial tier at 100 percent", abs(tier.loc["arterial", "coverage_pct"] - 100.0) < 1e-6)
    check("local tier near 25 percent", abs(tier.loc["local", "coverage_pct"] - 25.0) < 3.0)
    check("latest year carried through", seg.loc[0, "latest_year"] == 2023)

    print("\nOK" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
