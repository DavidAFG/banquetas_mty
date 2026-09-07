"""Who captured this city's imagery, and with what.

Stage 0 showed one organisation contributed 94 percent of Monterrey's images.
Knowing who they are matters for two reasons: they can be asked for the camera
mounting height, which the width estimation needs and Mapillary does not
publish, and they should be credited.
"""
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banquetas.config import load_city, mapillary_token

GRAPH = "https://graph.mapillary.com"


def lookup(endpoint_id, fields, token):
    try:
        r = requests.get(f"{GRAPH}/{endpoint_id}",
                         params={"access_token": token, "fields": fields}, timeout=30)
        return r.json() if r.status_code == 200 else {"error": f"{r.status_code} {r.text[:120]}"}
    except Exception as exc:
        return {"error": str(exc)}


def main(city_name="monterrey"):
    city = load_city(city_name)
    token = mapillary_token()

    pts = gpd.read_file(city.interim_dir / "mapillary_points.gpkg", layer="images",
                        columns=["organization_id", "creator_id"])
    for col, fields in (("organization_id", "id,slug,name,description"),
                        ("creator_id", "id,username")):
        if col not in pts.columns:
            continue
        top = pts[col].dropna().value_counts().head(3)
        print(f"\n{col}")
        for val, n in top.items():
            ident = str(int(float(val)))
            print(f"  {ident}  {n:,} images")
            print(f"    {lookup(ident, fields, token)}")

    meta_path = city.out_dir / "image_metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        if "camera_type" in meta:
            print("\ncamera types:", dict(meta["camera_type"].value_counts()))
        if "camera_parameters" in meta:
            print("\nmost common camera_parameters (focal, k1, k2):")
            for v, n in Counter(meta["camera_parameters"].dropna()).most_common(5):
                print(f"  {v}   {n:,} images")
    print("\nThe on screen watermark on the frames reads BLACKVUE DR900M-1CH/UHD,")
    print("a windscreen mounted consumer dashcam, so height is a car windscreen,")
    print("roughly 1.2 to 1.4 m. Worth confirming with the contributor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
