"""Test whether low coverage at the top of the income distribution is gating.

The hypothesis: coverage falls in the richest AGEB not because a mapping
operator chose to skip them, but because the streets are inside gated
developments that a vehicle cannot enter.

Two independent signals, deliberately kept separate so they can disagree:

1. Topology, computed here with no extra data. A gated fraccionamiento is a
   connected block of residential streets that touches the through network at
   very few points. Union find over shared endpoints gives the components;
   counting nodes shared with arterial or collector edges gives the entries.

2. OSM access tags, which osmnx already acted on. The `drive` network filter
   excludes ways tagged access=private, so explicitly gated streets are
   ALREADY missing from our denominator. scripts/private_network.py refetches
   with private ways included and measures the difference, which is the second
   signal and also a measurement of how much of the metro is private street
   space.

If both point the same way, the gating story holds. If only the topological
one does, what we are seeing is cul de sac layout rather than gating, which
is a different claim.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CityConfig

SNAP = 0.5          # metres, endpoint rounding for node identity
MIN_ENCLAVE_M = 300  # ignore tiny stubs
MAX_ENTRIES = 2      # a through street has many entries, an enclave has one or two
THROUGH_TIERS = {"arterial", "collector"}


class DSU:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _node(x: float, y: float) -> tuple[int, int]:
    return (int(round(x / SNAP)), int(round(y / SNAP)))


def components(edges: gpd.GeoDataFrame, crs) -> pd.DataFrame:
    """Connected components of the non through network, with entry counts."""
    e = edges.to_crs(crs)
    ends = []
    for row in e.itertuples(index=False):
        c = list(row.geometry.coords)
        ends.append((row.seg_id, row.tier, _node(*c[0][:2]), _node(*c[-1][:2]),
                     row.geometry.length))
    ends = pd.DataFrame(ends, columns=["seg_id", "tier", "a", "b", "length_m"])

    local = ends[~ends["tier"].isin(THROUGH_TIERS)]
    through_nodes = set(ends[ends["tier"].isin(THROUGH_TIERS)]["a"]) | \
                    set(ends[ends["tier"].isin(THROUGH_TIERS)]["b"])

    dsu = DSU()
    for r in local.itertuples(index=False):
        dsu.union(r.a, r.b)

    comp_of_seg, comp_nodes = {}, {}
    for r in local.itertuples(index=False):
        c = dsu.find(r.a)
        comp_of_seg[r.seg_id] = c
        comp_nodes.setdefault(c, set()).update((r.a, r.b))

    rows = []
    lens = local.groupby(local["seg_id"].map(comp_of_seg))["length_m"].sum()
    for c, nodes in comp_nodes.items():
        entries = len(nodes & through_nodes)
        rows.append({"comp": c, "n_nodes": len(nodes), "n_entries": entries,
                     "length_m": float(lens.get(c, 0.0))})
    comps = pd.DataFrame(rows)
    comps["enclave"] = (comps["length_m"] >= MIN_ENCLAVE_M) & (comps["n_entries"] <= MAX_ENTRIES)
    seg_map = pd.DataFrame({"seg_id": list(comp_of_seg), "comp": list(comp_of_seg.values())})
    return comps, seg_map


def run(city: CityConfig, verbose: bool = True) -> dict:
    edges = gpd.read_file(city.interim_dir / "osm_edges.gpkg", layer="edges")
    crs = city.crs_metric or edges.estimate_utm_crs()
    comps, seg_map = components(edges, crs)

    n_enc = int(comps["enclave"].sum())
    km_local = comps["length_m"].sum() / 1000
    km_enc = comps.loc[comps["enclave"], "length_m"].sum() / 1000
    if verbose:
        print(f"  non through components: {len(comps):,}")
        print(f"  classified as enclaves: {n_enc:,}  "
              f"({km_enc:,.0f} km of {km_local:,.0f} km, {100*km_enc/km_local:.1f}%)")
        print(f"  median enclave length: {comps.loc[comps['enclave'], 'length_m'].median():,.0f} m")

    ch = gpd.read_file(city.out_dir / "chunks.gpkg", layer="chunks",
                       columns=["chunk_id", "seg_id", "tier", "length_m", "n_images"])
    urb = gpd.read_file(city.out_dir / "chunks_urban.gpkg", layer="chunks",
                        columns=["chunk_id", "CVEGEO", "ses", "ses_decile", "mun_name"])
    df = pd.DataFrame(ch.drop(columns="geometry")).merge(
        pd.DataFrame(urb.drop(columns="geometry")), on="chunk_id", how="inner")
    df = df.merge(seg_map, on="seg_id", how="left").merge(
        comps[["comp", "enclave", "n_entries", "length_m"]].rename(
            columns={"length_m": "comp_length_m"}), on="comp", how="left")
    df["enclave"] = df["enclave"].fillna(False).astype(bool)
    df["observed"] = df["n_images"] > 0
    df["obs_len"] = df["length_m"] * df["observed"]

    def agg(by):
        g = df.groupby(by, dropna=False).agg(
            km=("length_m", lambda s: s.sum() / 1000),
            km_obs=("obs_len", lambda s: s.sum() / 1000)).reset_index()
        g["coverage_pct"] = 100 * g["km_obs"] / g["km"].replace(0, np.nan)
        return g

    local = df[~df["tier"].isin(THROUGH_TIERS)]

    if verbose:
        print("\nCoverage of residential streets, enclave versus through fabric")
        print(agg("enclave").to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

        print("\nShare of residential street length inside enclaves, by decile")
        L = local.copy()
        L["enc_len"] = L["length_m"] * L["enclave"]
        L["out_len"] = L["length_m"] * (~L["enclave"])
        L["out_obs"] = L["obs_len"] * (~L["enclave"])
        t = L.groupby("ses_decile", dropna=False).agg(
            km_=("length_m", "sum"), enc_=("enc_len", "sum"),
            obs_=("obs_len", "sum"), outl_=("out_len", "sum"), outo_=("out_obs", "sum"),
        ).reset_index()
        t["km"] = t["km_"] / 1000
        t["pct_enclave"] = 100 * t["enc_"] / t["km_"]
        t["coverage_all"] = 100 * t["obs_"] / t["km_"]
        t["coverage_excl_enclave"] = 100 * t["outo_"] / t["outl_"].replace(0, np.nan)
        t = t[["ses_decile", "km", "pct_enclave", "coverage_all", "coverage_excl_enclave"]]
        print(t.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

        obs = df.dropna(subset=["ses"])
        for label, sub in (("all urban chunks", obs),
                           ("excluding enclaves", obs[~obs["enclave"]])):
            s, o = sub["ses"].to_numpy(float), sub["observed"].to_numpy(float)
            slope = np.cov(s, o, bias=True)[0, 1] / s.var()
            print(f"  slope of observed on ses, {label}: {slope:+.4f}  (n={len(sub):,})")

    df.to_csv(city.out_dir / "chunks_enclave_flags.csv", index=False)
    comps.to_csv(city.out_dir / "enclave_components.csv", index=False)
    return {"components": comps, "chunks": df}
