"""Command line entry point.

    python -m banquetas coverage --city monterrey
    python -m banquetas coverage --city monterrey --areas data/raw/monterrey/ageb.gpkg --area-col CVEGEO
"""
from __future__ import annotations

import argparse
import sys

import geopandas as gpd

from . import areas as areas_mod
from . import enclaves as enclaves_mod
from . import images as images_mod
from . import sample as sample_mod
from . import measure as measure_mod
from . import segment as segment_mod
from . import coverage as coverage_mod
from . import mapillary, osm
from .config import load_city


def cmd_coverage(args) -> int:
    city = load_city(args.city)
    print(f"City: {city.label}  bbox={city.bbox}")

    print("\n[1/3] Road network")
    edges_path = city.interim_dir / "osm_edges.gpkg"
    if edges_path.exists() and not args.refresh:
        edges = gpd.read_file(edges_path, layer="edges")
        print(f"  reusing {len(edges)} segments from cache (pass --refresh to refetch)")
    else:
        edges = osm.road_network(city)

    print("\n[2/3] Mapillary image points")
    pts_path = city.interim_dir / "mapillary_points.gpkg"
    if pts_path.exists() and not args.refresh:
        points = gpd.read_file(pts_path, layer="images")
        print(f"  reusing {len(points)} image points from cache")
    else:
        points = mapillary.harvest(city)

    print("\n[3/3] Coverage")
    areas, area_col = None, None
    if args.areas:
        areas = gpd.read_file(args.areas)
        area_col = args.area_col
        if area_col not in areas.columns:
            print(f"  column '{area_col}' not in {args.areas}: {list(areas.columns)}", file=sys.stderr)
            return 2
    coverage_mod.run(city, edges, points, areas=areas, area_id_col=area_col,
                     chunk_m=args.chunk_m, snap_m=args.snap_m)
    print(f"\nOutputs in {city.out_dir}")
    return 0


def cmd_zones(args) -> int:
    """Stage 0 second pass: urban clip, municipality and AGEB join, SES gradient."""
    city = load_city(args.city)
    print(f"City: {city.label}")
    if not (city.out_dir / "chunks.gpkg").exists():
        print("Run the coverage command first.", file=sys.stderr)
        return 2
    areas_mod.run(city)
    print(f"\nOutputs in {city.out_dir}")
    return 0


def cmd_enclaves(args) -> int:
    """Test the gating hypothesis behind low coverage in the richest AGEB."""
    city = load_city(args.city)
    print(f"City: {city.label}")
    for f in ("chunks.gpkg", "chunks_urban.gpkg"):
        if not (city.out_dir / f).exists():
            print(f"{f} missing, run the coverage and zones commands first", file=sys.stderr)
            return 2
    enclaves_mod.run(city)
    print(f"\nOutputs in {city.out_dir}")
    return 0


def cmd_sample(args) -> int:
    city = load_city(args.city)
    print(f"City: {city.label}")
    sample_mod.run(city, n_target=args.n, seed=args.seed,
                   allocation=args.allocation, out_name=args.out)
    return 0


def cmd_images(args) -> int:
    city = load_city(args.city)
    print(f"City: {city.label}")
    if not (city.out_dir / "sample_chunks.gpkg").exists():
        print("Run the sample command first.", file=sys.stderr)
        return 2
    images_mod.run(city, k=args.k, dry_run=args.dry_run)
    return 0


def cmd_segment(args) -> int:
    city = load_city(args.city)
    print(f"City: {city.label}")
    if not (city.out_dir / "image_metadata.csv").exists():
        print("Run the images command first.", file=sys.stderr)
        return 2
    segment_mod.run(city, model_id=args.model, max_size=args.max_size, limit=args.limit,
                    overlays=args.overlays, refresh=args.refresh)
    return 0


def cmd_measure(args) -> int:
    city = load_city(args.city)
    print(f"City: {city.label}")
    measure_mod.run(city, limit=args.limit)
    print(f"\nOutputs in {city.out_dir}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="banquetas")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("coverage", help="Stage 0: measure how much of the network is observable")
    c.add_argument("--city", required=True)
    c.add_argument("--areas", help="Optional polygon layer to aggregate by (AGEB, municipio, colonia)")
    c.add_argument("--area-col", default="CVEGEO")
    c.add_argument("--chunk-m", type=float, default=coverage_mod.DEFAULT_CHUNK_M)
    c.add_argument("--snap-m", type=float, default=coverage_mod.DEFAULT_SNAP_M)
    c.add_argument("--refresh", action="store_true", help="Refetch instead of using cached inputs")
    c.set_defaults(func=cmd_coverage)

    z = sub.add_parser("zones", help="Stage 0 second pass: clip to urban AGEB and test the gradient")
    z.add_argument("--city", required=True)
    z.set_defaults(func=cmd_zones)

    en = sub.add_parser("enclaves", help="Topological test for gated developments")
    en.add_argument("--city", required=True)
    en.set_defaults(func=cmd_enclaves)

    sp = sub.add_parser("sample", help="Stage 1a: stratified sample of chunks to measure")
    sp.add_argument("--city", required=True)
    sp.add_argument("--n", type=int, default=2000)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--allocation", choices=("proportional", "equal"), default="proportional",
                    help="proportional for a pilot, equal for a validation set")
    sp.add_argument("--out", default="sample_chunks.gpkg")
    sp.set_defaults(func=cmd_sample)

    im = sub.add_parser("images", help="Stage 1a: select and download images for the sample")
    im.add_argument("--city", required=True)
    im.add_argument("--k", type=int, default=3, help="images per chunk")
    im.add_argument("--dry-run", action="store_true")
    im.set_defaults(func=cmd_images)

    sg = sub.add_parser("segment", help="Stage 1b: run semantic segmentation over the fetched images")
    sg.add_argument("--city", required=True)
    sg.add_argument("--model", default=segment_mod.DEFAULT_MODEL)
    sg.add_argument("--max-size", type=int, default=1024)
    sg.add_argument("--limit", type=int, help="stop after N images, for a trial run")
    sg.add_argument("--overlays", type=int, default=0,
                    help="write N side by side original and mask images for inspection")
    sg.add_argument("--refresh", action="store_true", help="recompute cached masks")
    sg.set_defaults(func=cmd_segment)

    ms = sub.add_parser("measure", help="Stage 1c: metres from the cached label maps")
    ms.add_argument("--city", required=True)
    ms.add_argument("--limit", type=int)
    ms.set_defaults(func=cmd_measure)

    args = p.parse_args(argv)
    return args.func(args)
