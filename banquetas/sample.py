"""Stage 1a, part one: choose which chunks to measure.

The audit universe is the observed arterial and collector network inside
urban AGEB, because stage 0 showed coverage there is high and close to
income neutral. Local streets are a separate, biased sample and are handled
later with an explicit selection model, not folded in here.

Sampling is stratified by municipality, socioeconomic decile and road tier
so the pilot cannot accidentally become a study of central Monterrey.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CityConfig

AUDIT_TIERS = ("arterial", "collector")

# Pedestrians are prohibited on these, so a missing sidewalk is not a finding.
# They are excluded from the audit universe rather than counted as failures.
NON_PEDESTRIAN = ("motorway", "motorway_link", "trunk", "trunk_link")
STRATA = ("mun_name", "ses_decile", "tier")


def build_universe(city: CityConfig, tiers=AUDIT_TIERS, verbose: bool = True) -> gpd.GeoDataFrame:
    g = gpd.read_file(city.out_dir / "chunks_urban.gpkg", layer="chunks")
    before = len(g)
    g = g[g["tier"].isin(tiers) & (g["n_images"] > 0)].copy()
    if "highway" not in g.columns:
        raise RuntimeError(
            "chunks_urban.gpkg has no highway column, so motorways and trunk roads "
            "cannot be excluded. Rerun `zones` first; it now carries the tag.")
    g = g[~g["highway"].isin(NON_PEDESTRIAN)]
    if verbose:
        print(f"  dropped {before - len(g):,} chunks: unobserved, wrong tier, "
              f"or motorway and trunk where pedestrians are prohibited")
    if verbose:
        km = g["length_m"].sum() / 1000
        print(f"  audit universe: {len(g):,} observed chunks, {km:,.0f} km, tiers {', '.join(tiers)}")
        print(g.groupby("tier")["length_m"].agg(chunks="size", km=lambda s: s.sum() / 1000)
              .to_string(float_format=lambda v: f"{v:,.0f}"))
    return g


def stratified(universe: gpd.GeoDataFrame, n_target: int, seed: int = 0,
               min_per_stratum: int = 2, allocation: str = "proportional",
               verbose: bool = True) -> gpd.GeoDataFrame:
    """Draw a stratified sample.

    allocation="proportional" mirrors the universe, which is what you want for
    a pilot meant to surface typical failure modes.

    allocation="equal" gives every stratum the same target, which is what you
    want for a validation set. The universe is skewed towards richer AGEB
    because coverage is, so a proportional validation sample would measure
    accuracy mostly where the streets are widest and best built, and quietly
    overstate it everywhere else.
    """
    rng = np.random.default_rng(seed)
    u = universe.copy()
    parts = [u[c].astype("object").fillna("NA").astype(str) for c in STRATA]
    u["_stratum"] = parts[0].str.cat(parts[1:], sep="|")

    sizes = u["_stratum"].value_counts()
    if allocation == "equal":
        alloc = pd.Series(max(1, round(n_target / len(sizes))), index=sizes.index)
    elif allocation == "proportional":
        alloc = (sizes / sizes.sum() * n_target).round().astype(int)
    else:
        raise ValueError(f"unknown allocation {allocation!r}")
    alloc = np.maximum(alloc, min_per_stratum).clip(upper=sizes)

    picks = []
    for stratum, k in alloc.items():
        idx = u.index[u["_stratum"] == stratum].to_numpy()
        picks.append(rng.choice(idx, size=int(min(k, len(idx))), replace=False))
    sample = u.loc[np.concatenate(picks)].drop(columns="_stratum")

    if verbose:
        print(f"  sampled {len(sample):,} chunks across {len(sizes):,} strata "
              f"(target {n_target:,})")
        print(sample.groupby("tier").size().to_string())
        d = sample.groupby("ses_decile", dropna=False).size()
        print("  by decile: " + ", ".join(f"{k}:{v}" for k, v in d.items()))
    return sample


def run(city: CityConfig, n_target: int = 2000, seed: int = 0,
        tiers=AUDIT_TIERS, allocation: str = "proportional",
        out_name: str = "sample_chunks.gpkg", verbose: bool = True) -> gpd.GeoDataFrame:
    universe = build_universe(city, tiers=tiers, verbose=verbose)
    sample = stratified(universe, n_target, seed=seed, allocation=allocation, verbose=verbose)
    out = city.out_dir / out_name
    sample.to_file(out, driver="GPKG", layer="sample")
    if verbose:
        print(f"  wrote {out}")
    return sample
