"""Stage 0, second pass: clip to real urban geography and test the coverage gradient.

Replaces the rectangular bounding box with INEGI urban AGEB, attaches
municipality and AGEB identifiers to every chunk, joins census
sociodemographics, and reports coverage by municipality and by
socioeconomic decile.

Inputs expected under data/raw/<city>/
  mg2020/            shapefiles from the Marco Geoestadistico, state layers
  censo_ageb_mza/    the RESAGEBURB csv for the state
Both are fetched by fetch_inegi.sh.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import CityConfig


# ---------------------------------------------------------------- discovery

def discover_layers(mg_dir: Path) -> dict[str, Path]:
    """Identify the AGEB and municipio layers without hardcoding file names.

    INEGI naming has changed across editions, so layers are identified by the
    columns they carry and by geometry type.
    """
    found: dict[str, Path] = {}
    for shp in sorted(mg_dir.glob("*.shp")):
        try:
            head = gpd.read_file(shp, rows=1)
        except Exception:
            continue
        cols = {c.upper() for c in head.columns}
        gtype = head.geometry.geom_type.iloc[0] if len(head) else ""
        if "Polygon" not in str(gtype):
            continue
        if {"CVE_AGEB"} <= cols and "ageb" not in found:
            found["ageb"] = shp
        elif {"CVE_MUN"} <= cols and not {"CVE_AGEB", "CVE_LOC"} & cols and "mun" not in found:
            found["mun"] = shp
    return found


def report_inputs(city: CityConfig) -> dict[str, Path]:
    mg_dir = city.raw_dir / "mg2020"
    if not mg_dir.exists():
        raise FileNotFoundError(f"{mg_dir} not found. Run fetch_inegi.sh first.")
    shp = sorted(mg_dir.glob("*.shp"))
    print(f"  shapefiles in {mg_dir.name}: {', '.join(p.stem for p in shp) or '(none)'}")
    layers = discover_layers(mg_dir)
    for k, v in layers.items():
        print(f"    {k:5s} -> {v.name}")
    missing = {"ageb", "mun"} - set(layers)
    if missing:
        raise RuntimeError(f"Could not identify layer(s): {', '.join(sorted(missing))}. "
                           f"Inspect the shapefiles above and pass them explicitly.")
    return layers


# ---------------------------------------------------------------- loading

def load_ageb(city: CityConfig, path: Path) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    g.columns = [c.upper() if c != "geometry" else c for c in g.columns]
    if "CVEGEO" not in g.columns:
        g["CVEGEO"] = (g["CVE_ENT"].astype(str).str.zfill(2) + g["CVE_MUN"].astype(str).str.zfill(3)
                       + g["CVE_LOC"].astype(str).str.zfill(4) + g["CVE_AGEB"].astype(str).str.zfill(4))
    codes = set(city.inegi.get("municipality_codes", []))
    if codes:
        g = g[g["CVE_MUN"].astype(str).str.zfill(3).isin(codes)]
    print(f"  urban AGEB in the metro municipalities: {len(g):,}")
    return g


def load_mun(city: CityConfig, path: Path) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    g.columns = [c.upper() if c != "geometry" else c for c in g.columns]
    codes = set(city.inegi.get("municipality_codes", []))
    if codes:
        g = g[g["CVE_MUN"].astype(str).str.zfill(3).isin(codes)]
    name_col = next((c for c in ("NOMGEO", "NOM_MUN", "NOMBRE") if c in g.columns), None)
    g["mun_name"] = g[name_col] if name_col else g["CVE_MUN"]
    return g


CENSUS_VARS = {
    "POBTOT": "pop",
    "TVIVPARHAB": "dwellings",
    "VPH_INTER": "vph_internet",
    "VPH_AUTOM": "vph_car",
    "VPH_PISOTI": "vph_dirt_floor",
    "GRAPROES": "mean_schooling",
    "PRO_OCUP_C": "occupants_per_room",
}


def load_census(city: CityConfig) -> pd.DataFrame:
    d = city.raw_dir / "censo_ageb_mza"
    files = list(d.rglob("*.csv") ) + list(d.rglob("*.CSV"))
    if not files:
        raise FileNotFoundError(f"No csv under {d}. Run fetch_inegi.sh first.")
    path = max(files, key=lambda p: p.stat().st_size)
    print(f"  census file: {path.name}")
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, encoding="latin-1", low_memory=False)
    df.columns = [c.replace("\ufeff", "").upper().strip() for c in df.columns]

    # AGEB level rows only: a real AGEB code and the manzana field at zero
    df = df[(df["AGEB"].str.strip() != "0000") & (df["MZA"].astype(str).str.strip().isin(["0", "000", "0000"]))]
    df["CVEGEO"] = (df["ENTIDAD"].str.zfill(2) + df["MUN"].str.zfill(3)
                    + df["LOC"].str.zfill(4) + df["AGEB"].str.zfill(4))

    keep = {}
    for src, dst in CENSUS_VARS.items():
        if src in df.columns:
            keep[dst] = pd.to_numeric(df[src].replace({"*": np.nan, "N/D": np.nan, "": np.nan}),
                                      errors="coerce")
    out = pd.DataFrame(keep)
    out["CVEGEO"] = df["CVEGEO"].values
    print(f"  census AGEB rows: {len(out):,}  variables: {', '.join(keep)}")
    return out


def ses_index(df: pd.DataFrame) -> pd.Series:
    """Transparent socioeconomic index: z scores of a few census shares, averaged.

    Higher is better off. Deliberately simple and inspectable rather than a
    latent factor, because it will be argued about.
    """
    parts = []
    d = df["dwellings"].replace(0, np.nan)
    if "vph_internet" in df:
        parts.append(("internet_share", df["vph_internet"] / d))
    if "vph_car" in df:
        parts.append(("car_share", df["vph_car"] / d))
    if "mean_schooling" in df:
        parts.append(("mean_schooling", df["mean_schooling"]))
    if "occupants_per_room" in df:
        parts.append(("crowding_inv", -df["occupants_per_room"]))
    if "vph_dirt_floor" in df:
        parts.append(("dirt_floor_inv", -(df["vph_dirt_floor"] / d)))
    if not parts:
        raise RuntimeError("No census variables available for the index")
    z = pd.DataFrame({name: (s - s.mean()) / s.std(ddof=0) for name, s in parts})
    print(f"  SES index built from: {', '.join(z.columns)}")
    return z.mean(axis=1, skipna=True)


# ---------------------------------------------------------------- main

def run(city: CityConfig, verbose: bool = True) -> dict:
    layers = report_inputs(city)
    ageb = load_ageb(city, layers["ageb"])
    mun = load_mun(city, layers["mun"])
    census = load_census(city)

    census["ses"] = ses_index(census)
    census["ses_decile"] = pd.qcut(census["ses"].rank(method="first"), 10,
                                   labels=[f"D{i}" for i in range(1, 11)])

    want = ["chunk_id", "seg_id", "tier", "highway", "length_m", "n_images"]
    try:
        chunks = gpd.read_file(city.out_dir / "chunks.gpkg", layer="chunks", columns=want)
    except Exception:
        chunks = gpd.read_file(city.out_dir / "chunks.gpkg", layer="chunks")
    print(f"  chunks loaded: {len(chunks):,}")

    mids = chunks[["chunk_id", "geometry"]].copy()
    mids["geometry"] = mids.geometry.interpolate(0.5, normalized=True)

    a = ageb.to_crs(chunks.crs)[["CVEGEO", "CVE_MUN", "geometry"]]
    j = gpd.sjoin(mids, a, how="left", predicate="within").drop_duplicates("chunk_id")
    chunks = chunks.merge(j[["chunk_id", "CVEGEO", "CVE_MUN"]], on="chunk_id", how="left")

    inside = chunks["CVEGEO"].notna()
    print(f"  chunks inside urban AGEB: {inside.sum():,} of {len(chunks):,} "
          f"({100 * inside.mean():.1f}%)")
    urban = chunks[inside].copy()
    urban["observed"] = urban["n_images"] > 0
    urban["obs_len"] = urban["length_m"] * urban["observed"].astype(int)

    munname = mun[["CVE_MUN", "mun_name"]].drop_duplicates()
    urban = urban.merge(munname, on="CVE_MUN", how="left")
    urban = urban.merge(census[["CVEGEO", "ses", "ses_decile"]], on="CVEGEO", how="left")

    def agg(by):
        g = urban.groupby(by, dropna=False).agg(
            km_total=("length_m", lambda s: s.sum() / 1000),
            km_observed=("obs_len", lambda s: s.sum() / 1000),
            n_chunks=("chunk_id", "size"),
            n_images=("n_images", "sum"),
        ).reset_index()
        g["coverage_pct"] = 100 * g["km_observed"] / g["km_total"].replace(0, np.nan)
        return g

    by_mun = agg("mun_name").sort_values("coverage_pct", ascending=False)
    by_dec = agg("ses_decile")
    by_dec_tier = agg(["ses_decile", "tier"])

    urban.to_file(city.out_dir / "chunks_urban.gpkg", driver="GPKG", layer="chunks")
    by_mun.to_csv(city.out_dir / "coverage_by_municipality.csv", index=False)
    by_dec.to_csv(city.out_dir / "coverage_by_ses_decile.csv", index=False)
    by_dec_tier.to_csv(city.out_dir / "coverage_by_ses_decile_tier.csv", index=False)

    if verbose:
        fmt = lambda v: f"{v:,.1f}"
        print("\nCoverage by municipality")
        print(by_mun.to_string(index=False, float_format=fmt))
        print("\nCoverage by socioeconomic decile (D1 poorest, D10 best off)")
        print(by_dec.to_string(index=False, float_format=fmt))
        obs = urban.dropna(subset=["ses"])
        if len(obs) > 1000:
            r = np.corrcoef(obs["ses"], obs["observed"].astype(float))[0, 1]
            print(f"\nChunk level correlation between AGEB socioeconomic index and being observed: {r:+.3f}")

    return {"by_mun": by_mun, "by_decile": by_dec, "by_decile_tier": by_dec_tier, "urban": urban}
