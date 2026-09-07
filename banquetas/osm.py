"""Road network retrieval and hierarchy classification.

osmnx is used when available because it handles the Overpass paging and the
graph cleanup. If it is missing, a plain Overpass query is used instead, so
the pipeline still runs on a bare environment.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from .config import CityConfig

OVERPASS = "https://overpass-api.de/api/interpreter"


def _first(value):
    """OSM tags are sometimes lists after osmnx merges parallel ways."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def road_network(city: CityConfig, verbose: bool = True,
                 out_name: str = "osm_edges.gpkg",
                 custom_filter: str | None = None) -> gpd.GeoDataFrame:
    west, south, east, north = city.bbox
    try:
        import osmnx as ox
    except ImportError:
        return _road_network_overpass(city, verbose=verbose, out_name=out_name)

    network_type = city.osm.get("network_type", "drive")
    n_tiles = int(city.osm.get("tiles", 3))  # split the bbox so Overpass does not time out
    major = int(ox.__version__.split(".")[0]) if hasattr(ox, "__version__") else 2

    def one_bbox(w, s, e, n):
        kw = {"simplify": True, "retain_all": True}
        if custom_filter:
            kw["custom_filter"] = custom_filter
        else:
            kw["network_type"] = network_type
        if major >= 2:
            g = ox.graph_from_bbox((w, s, e, n), **kw)
        else:
            g = ox.graph_from_bbox(n, s, e, w, **kw)
        return ox.graph_to_gdfs(g, nodes=False, edges=True).reset_index()

    if verbose:
        print(f"  querying OSM through osmnx {getattr(ox, '__version__', '?')} "
              f"in {n_tiles}x{n_tiles} tiles ...", flush=True)

    lon_edges = [west + (east - west) * i / n_tiles for i in range(n_tiles + 1)]
    lat_edges = [south + (north - south) * i / n_tiles for i in range(n_tiles + 1)]
    parts, k = [], 0
    for i in range(n_tiles):
        for j in range(n_tiles):
            k += 1
            try:
                part = one_bbox(lon_edges[i], lat_edges[j], lon_edges[i + 1], lat_edges[j + 1])
                parts.append(part)
                if verbose:
                    print(f"    tile {k}/{n_tiles * n_tiles}: {len(part)} edges", flush=True)
            except Exception as exc:  # an empty tile raises in some osmnx versions
                if verbose:
                    print(f"    tile {k}/{n_tiles * n_tiles}: skipped "
                          f"({type(exc).__name__}: {exc})", flush=True)

    if not parts:
        raise RuntimeError("OSM returned nothing for this bbox")
    edges = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    if "osmid" in edges.columns:
        edges["_key"] = edges["osmid"].map(lambda v: str(v)) + "|" + edges.geometry.to_wkb().map(hash).astype(str)
        edges = edges.drop_duplicates(subset="_key").drop(columns="_key")
    return _tidy(edges, city, verbose=verbose, out_name=out_name)


def _road_network_overpass(city: CityConfig, verbose: bool = True,
                           out_name: str = "osm_edges.gpkg") -> gpd.GeoDataFrame:
    import json

    import requests
    from shapely.geometry import LineString

    west, south, east, north = city.bbox
    query = f"""
    [out:json][timeout:180];
    way["highway"]["highway"!~"footway|path|steps|cycleway|track|service|construction|proposed"]
      ({south},{west},{north},{east});
    (._;>;);
    out body;
    """
    if verbose:
        print("  querying Overpass directly (osmnx not installed) ...", flush=True)
    r = requests.post(OVERPASS, data={"data": query}, timeout=300)
    r.raise_for_status()
    payload = json.loads(r.text)
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in payload["elements"] if e["type"] == "node"}
    rows = []
    for e in payload["elements"]:
        if e["type"] != "way":
            continue
        coords = [nodes[n] for n in e["nodes"] if n in nodes]
        if len(coords) < 2:
            continue
        tags = e.get("tags", {})
        rows.append({
            "osmid": e["id"],
            "highway": tags.get("highway"),
            "name": tags.get("name"),
            "maxspeed": tags.get("maxspeed"),
            "sidewalk": tags.get("sidewalk"),
            "geometry": LineString(coords),
        })
    edges = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return _tidy(edges, city, verbose=verbose, out_name=out_name)


def _tidy(edges: gpd.GeoDataFrame, city: CityConfig, verbose: bool = True,
          out_name: str = "osm_edges.gpkg") -> gpd.GeoDataFrame:
    keep = [c for c in ("osmid", "highway", "name", "maxspeed", "sidewalk",
                        "access", "barrier", "service", "geometry") if c in edges.columns]
    edges = edges[keep].copy()
    for col in ("highway", "name", "maxspeed", "sidewalk", "access", "barrier", "service", "osmid"):
        if col in edges.columns:
            edges[col] = edges[col].map(_first)

    edges["highway"] = edges["highway"].astype("string")
    edges["tier"] = edges["highway"].map(city.tier_of)
    edges = edges[edges.geometry.notna() & (edges.geometry.geom_type == "LineString")]
    edges = edges.reset_index(drop=True)
    edges["seg_id"] = edges.index.astype(int)

    # maxspeed arrives as "50", "50 km/h" or nothing at all
    edges["maxspeed_kph"] = (
        edges.get("maxspeed", pd.Series(index=edges.index, dtype="object"))
        .astype("string")
        .str.extract(r"(\d+)")[0]
        .astype("Float64")
    )

    out = city.interim_dir / out_name
    edges.to_file(out, driver="GPKG", layer="edges")
    if verbose:
        print(f"  wrote {len(edges)} road segments to {out}")
        print(edges["tier"].value_counts().to_string())
    return edges
