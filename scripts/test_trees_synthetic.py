"""Offline check for the street tree side module.

Three real trees, each seen from several passing images with noisy distance
estimates, must collapse into three candidate locations.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from banquetas import trees

LAT0, LON0 = 25.68, -100.30
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.radians(LAT0))

rng = np.random.default_rng(7)


def main():
    # three trees on the north side of an east west street, 30 m apart
    tree_x = [0.0, 30.0, 60.0]
    tree_y = 8.0
    observations = []

    for k, tx in enumerate(tree_x):
        # camera drives east along y = 0, sees the tree from three positions
        for cam_x in (tx - 12, tx, tx + 12):
            dx, dy = tx - cam_x, tree_y
            distance = float(np.hypot(dx, dy) + rng.normal(0, 0.8))
            bearing = float(np.degrees(np.arctan2(dx, dy)))  # relative to north
            row = pd.Series({
                "id": f"img_{k}_{cam_x:.0f}",
                "lon": LON0 + cam_x / M_PER_DEG_LON,
                "lat": LAT0,
                "compass_angle": 0.0,   # camera facing north for simplicity
                "year": 2023,
            })
            observations += trees.observations_from_image(
                row, [{"class_name": "vegetation", "bearing_offset_deg": bearing,
                       "distance_m": distance, "pixel_area": 5000}])

    gdf = trees.to_gdf(observations)
    clusters = trees.cluster(gdf, crs="EPSG:32614", eps_m=6.0, min_obs=2)

    print(f"observations: {len(gdf)}")
    print(clusters.drop(columns="geometry").to_string(index=False))

    ok = len(clusters) == 3 and set(clusters["n_obs"]) == {3}
    print("\nOK" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
