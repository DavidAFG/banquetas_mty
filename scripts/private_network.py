"""Second gating signal: how much drivable street length is tagged private.

osmnx's `drive` filter silently drops ways tagged access=private, so gated
streets never entered the stage 0 denominator. This refetches with exactly
the same filter MINUS that one exclusion, keeps the access tag, and measures
the private complement directly rather than by differencing two networks.

The result is a measurement in its own right: how much of the metro's street
space is privately governed, and where it sits in the income distribution.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from banquetas import areas, osm
from banquetas.config import load_city

# The osmnx drive filter, with ["access"!~"private"] removed.
DRIVE_INCLUDING_PRIVATE = (
    '["highway"]["area"!~"yes"]'
    '["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|'
    'elevator|escalator|footway|path|pedestrian|planned|platform|proposed|raceway|'
    'service|steps|track"]'
    '["motor_vehicle"!~"no"]["motor_car"!~"no"]'
    '["service"!~"alley|driveway|emergency_access|parking|private"]'
)
PRIVATE_VALUES = {"private", "no", "customers", "permit", "permissive"}


def main(city_name="monterrey"):
    city = load_city(city_name)
    public = gpd.read_file(city.interim_dir / "osm_edges.gpkg", layer="edges")

    print("fetching the drive network including private ways, this takes a while")
    allways = osm.road_network(city, verbose=True,
                               out_name="osm_edges_incl_private.gpkg",
                               custom_filter=DRIVE_INCLUDING_PRIVATE)

    crs = city.crs_metric or public.estimate_utm_crs()
    a = allways.to_crs(crs)
    a["length_m"] = a.length
    acc = a["access"].astype("string").str.lower() if "access" in a.columns else pd.Series(index=a.index, dtype="string")
    a["is_private"] = acc.isin(PRIVATE_VALUES).fillna(False)

    km_pub = public.to_crs(crs).length.sum() / 1000
    km_all = a["length_m"].sum() / 1000
    km_priv = a.loc[a["is_private"], "length_m"].sum() / 1000
    print(f"\npublic drive network (stage 0):  {km_pub:>9,.0f} km")
    print(f"including private ways:          {km_all:>9,.0f} km")
    print(f"tagged private:                  {km_priv:>9,.0f} km  "
          f"({100 * km_priv / km_all:.1f}% of drivable street length)")
    if "access" in a.columns:
        print("\naccess tag values present:")
        print(a.groupby(acc.fillna("(untagged)"))["length_m"].sum().div(1000)
              .sort_values(ascending=False).head(8).to_string(float_format=lambda v: f"{v:,.1f} km"))

    # where does private street space sit in the income distribution
    layers = areas.report_inputs(city)
    ageb = areas.load_ageb(city, layers["ageb"]).to_crs(crs)
    census = areas.load_census(city)
    census["ses"] = areas.ses_index(census)
    census["ses_decile"] = pd.qcut(census["ses"].rank(method="first"), 10,
                                   labels=[f"D{i}" for i in range(1, 11)])

    mids = a[["length_m", "is_private", "geometry"]].copy()
    mids["geometry"] = mids.geometry.interpolate(0.5, normalized=True)
    j = gpd.sjoin(mids, ageb[["CVEGEO", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()].merge(census[["CVEGEO", "ses_decile"]], on="CVEGEO", how="left")

    j["priv_len"] = j["length_m"] * j["is_private"]
    t = j.groupby("ses_decile", dropna=False).agg(
        km=("length_m", lambda s: s.sum() / 1000),
        km_private=("priv_len", lambda s: s.sum() / 1000)).reset_index()
    t["pct_private"] = 100 * t["km_private"] / t["km"].replace(0, np.nan)
    print("\nPrivate street length by socioeconomic decile")
    print(t.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    t.to_csv(city.out_dir / "private_streets_by_decile.csv", index=False)
    print(f"\nwrote {city.out_dir / 'private_streets_by_decile.csv'}")
    print("osm_edges.gpkg, the public network stage 0 used, is untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
