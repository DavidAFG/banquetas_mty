"""Build a ground truth set for sidewalk width, measured in Google Earth.

Lane markings calibrate the camera height, but they cannot tell you whether
the sidewalk edge itself is being found in the right place. For that you need
a handful of widths measured independently. Google Earth's ruler on high
resolution imagery is free and good to a few centimetres at this scale.

Writes a CSV to fill in and a KML to open, with one placemark per block face,
nudged a few metres onto the side that was actually measured so there is no
ambiguity about which kerb to put the ruler on.

Faces are drawn across the whole range of estimated widths, not at random, so
the regression has leverage at both ends rather than clustering at the median.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banquetas.config import load_city

N_POINTS = 40
OFFSET_M = 5.0

KML_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<name>banquetas width check</name>
"""
KML_PT = """<Placemark><name>{name}</name>
<description>estimated {w:.2f} m, face bearing {b:.0f} deg, {n} images</description>
<Point><coordinates>{lon:.7f},{lat:.7f},0</coordinates></Point></Placemark>
"""


def main(city_name="monterrey", n=N_POINTS, seed=3):
    city = load_city(city_name)
    rng = np.random.default_rng(seed)

    faces = pd.read_csv(city.out_dir / "blockface_measurements.csv")
    faces = faces[faces["width_m"].notna() & (faces["n_images"] >= 2)]
    chunks = gpd.read_file(city.out_dir / "sample_chunks.gpkg", layer="sample")

    # stratify across the width distribution so the fit has leverage at the tails
    faces = faces.copy()
    faces["band"] = pd.qcut(faces["width_m"], min(8, faces["width_m"].nunique()),
                            duplicates="drop")
    per = max(1, n // faces["band"].nunique())
    picks = [g.iloc[rng.choice(len(g), size=min(per, len(g)), replace=False)]
             for _, g in faces.groupby("band", observed=True)]
    sel = pd.concat(picks).head(n)

    geo = chunks.set_index("chunk_id").loc[sel["chunk_id"]]
    mid = geo.geometry.interpolate(0.5, normalized=True)
    crs = chunks.crs

    # step off the centreline onto the measured side
    br = np.radians(sel["face_bearing"].to_numpy())
    dx, dy = np.sin(br) * OFFSET_M, np.cos(br) * OFFSET_M
    pts = gpd.GeoSeries(gpd.points_from_xy(mid.x.to_numpy() + dx, mid.y.to_numpy() + dy),
                        crs=crs).to_crs("EPSG:4326")

    out = pd.DataFrame({
        "chunk_id": sel["chunk_id"].to_numpy(),
        "face": sel["face"].to_numpy(),
        "lat": pts.y.to_numpy().round(7),
        "lon": pts.x.to_numpy().round(7),
        "face_bearing": sel["face_bearing"].round(0).to_numpy(),
        "n_images": sel["n_images"].to_numpy(),
        "estimated_width_m": sel["width_m"].round(2).to_numpy(),
        "measured_width_m": "",          # fill this in from Google Earth
        "notes": "",
    })
    d = city.out_dir / "width_check"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "width_check.csv", index=False)

    kml = [KML_HEAD]
    for r in out.itertuples(index=False):
        kml.append(KML_PT.format(name=f"{r.chunk_id}-{r.face}", w=r.estimated_width_m,
                                 b=r.face_bearing, n=r.n_images, lon=r.lon, lat=r.lat))
    kml.append("</Document></kml>\n")
    (d / "width_check.kml").write_text("".join(kml))

    print(f"  {len(out)} block faces across the width range")
    print(f"    estimated width: min {out['estimated_width_m'].min():.2f}, "
          f"median {out['estimated_width_m'].median():.2f}, "
          f"max {out['estimated_width_m'].max():.2f}")
    print(f"  open {d / 'width_check.kml'} in Google Earth")
    print(f"  fill measured_width_m in {d / 'width_check.csv'}")
    print("  measure kerb face to building line or back of walk, on the side the")
    print("  placemark sits on, then run scripts/score_width.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
