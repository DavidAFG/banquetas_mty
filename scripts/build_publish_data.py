"""Bundle everything a shareable version of the map needs into one file.

The published page cannot reach an external basemap: hosted artifacts block
fetches and tiles from anywhere but a short allowlist, and no tile server is
on it. So geographic context has to travel with the data. Municipality
outlines are enough to anchor 3,285 scattered block faces, and they are
cheaper and cleaner than a basemap for a figure anyway.

Coordinates are rounded to five decimals, about a metre, and the outlines are
simplified, because the whole bundle has to fit inside one HTML file.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banquetas.config import load_city
from banquetas import areas

SIMPLIFY_M = 25
PRECISION = 5


def round_coords(obj, nd=PRECISION):
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            return [round(float(v), nd) for v in obj]
        return [round_coords(o, nd) for o in obj]
    return obj


def main(city_name="monterrey"):
    city = load_city(city_name)

    faces = json.loads((city.out_dir / "map" / "data.geojson").read_text())
    for f in faces["features"]:
        f["geometry"]["coordinates"] = round_coords(f["geometry"]["coordinates"])
        p = f["properties"]
        for k in ("kerb_offset_m", "road_width_m", "present_rate", "rooted_frac",
                  "canopy_frac", "obstructed_frac"):
            p.pop(k, None)
    print(f"  {len(faces['features']):,} block faces")

    layers = areas.report_inputs(city)
    mun = areas.load_mun(city, layers["mun"])
    m = mun.to_crs(mun.estimate_utm_crs())
    m["geometry"] = m.geometry.simplify(SIMPLIFY_M).buffer(0)
    m = m.to_crs("EPSG:4326")[["mun_name", "geometry"]]
    mj = json.loads(m.to_json())
    for f in mj["features"]:
        f["geometry"]["coordinates"] = round_coords(f["geometry"]["coordinates"])
    print(f"  {len(mj['features'])} municipality outlines")

    # a label anchor per municipality, so names can be drawn without a basemap
    pts = m.copy()
    pts["geometry"] = pts.geometry.representative_point()
    lab = [{"name": r.mun_name, "lon": round(r.geometry.x, PRECISION),
            "lat": round(r.geometry.y, PRECISION)} for r in pts.itertuples(index=False)]

    out = city.out_dir / "map" / "publish.json"
    out.write_text(json.dumps({"faces": faces, "municipalities": mj, "labels": lab},
                              separators=(",", ":")))
    print(f"  wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
