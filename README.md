# banquetas

A city agnostic pipeline for auditing sidewalk provision and sidewalk quality in Mexican cities, using open street level imagery and official INEGI data.

INEGI already records, for every block face in the country, whether a sidewalk exists. This pipeline adds the layer that INEGI does not have: how wide the sidewalk is, whether it is obstructed, and whether it is shaded. Nothing in the code is specific to any one city. Adding a city means adding one YAML file.

## Why coverage comes first

Open street level imagery is not evenly distributed. In the Monterrey metropolitan area, visual inspection shows near complete coverage of local streets in San Pedro, partial coverage of fraccionamiento perimeters in Escobedo, and large contiguous residential areas of Guadalupe with nothing at all. Coverage correlates with income, in the direction that makes citywide sidewalk quality look better than it is and flattens the inequality gradient.

So stage 0 measures what can be seen before anything is measured about sidewalks. The answer determines the scope of everything that follows, and the map of what cannot be seen is itself a result worth publishing.

## Stages

| Stage | What it does | Status |
| --- | --- | --- |
| 0 | Coverage audit: how much of the network is observable, by road tier and by area | implemented |
| 1 | Sidewalk presence from imagery, validated against INEGI BANQUETA | not started |
| 2 | Quality: width, obstruction, trees | not started |
| 3 | Interactive audit map | not started |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Get a free client token from the [Mapillary developer dashboard](https://www.mapillary.com/dashboard/developers), then:

```bash
cp .env.example .env
# edit .env, then
source .env
```

The token is read from the `MAPILLARY_TOKEN` environment variable only. It is never written to disk by the code and `.env` is gitignored.

## Run stage 0

```bash
python -m banquetas coverage --city monterrey
```

This fetches the road network from OpenStreetMap, harvests Mapillary image point metadata from vector tiles, and reports coverage. No photographs are downloaded at this stage, only the metadata saying where photographs exist, so it is fast and it costs nothing against the rate limit that matters later.

To aggregate coverage by census geography, pass a polygon layer:

```bash
python -m banquetas coverage --city monterrey \
  --areas data/raw/monterrey/ageb.gpkg --area-col CVEGEO
```

Useful flags:

- `--chunk-m` length of the units coverage is measured on, default 25 m. A long avenue with images at one intersection should not count as covered, which is why coverage is not measured on whole OSM segments.
- `--snap-m` how far an image can be from a chunk and still count for it, default 12 m.
- `--refresh` refetch instead of reusing the cached network and image points.

Outputs land in `data/out/<city>/`:

- `chunks.gpkg` one row per chunk with image count and latest capture year
- `segments.gpkg` one row per OSM segment with a coverage fraction
- `coverage_by_tier.csv` and `coverage_by_area.csv`

Mapillary tiles are cached under `data/raw/<city>/mapillary_tiles/`, so an interrupted run resumes rather than refetching.

## Adding a city

Copy `config/cities/monterrey.yml`, change the name, the bounding box and the INEGI codes. Leave `crs_metric` null and the pipeline picks the correct local UTM zone by itself. Everything downstream is driven by that file.

## Data you will need for stage 1

These are manual downloads, once per state, all free:

- INEGI, Características del entorno urbano 2020. Block face level, with `BANQUETA`, `GUARNICION`, `RAMPAS`, `ARBOLES`, `ALUMPUB`, `PASOPEAT`, `SEMAFOROPEAT`.
- INEGI, cartography for AGEB and manzana, for aggregation and for sociodemographic joins.

Drop them in `data/raw/<city>/` and point the loaders at them.

## The street tree module

`banquetas/trees.py` is a deliberate side project, not part of the audit path. It rides along on the segmentation stage: vegetation detected at street level is projected to an approximate ground position and clustered across images into candidate tree locations. INEGI records only a binary for trees per block face, so even a rough point inventory is strictly more information, and street trees are the shade layer that connects this work to the heat island research.

Positions are approximate. Monocular distance estimates degrade with range and a hedge is not a tree. The output is candidate locations to be validated, never a cadastre. The clustering and projection are implemented and tested; the per image detection is stubbed with a documented interface until stage 2 wires in the segmentation model.

## Tests

Both run offline, with no token and no network:

```bash
python scripts/test_coverage_synthetic.py
python scripts/test_trees_synthetic.py
```

The first builds a synthetic grid with known coverage and checks that the geometry, the snapping and the aggregation return the right numbers. The second checks that repeated sightings of the same tree collapse into one candidate.

## Notes on method

- Coverage is length weighted, not segment counted, so a long unobserved avenue is not equal to a short unobserved cul de sac.
- Each image point is assigned to its nearest chunk within the snap distance, so an image at an intersection is counted once rather than once per touching segment.
- Imagery older than `mapillary.min_year` is dropped. Mixing a 2015 image with a 2025 one produces disagreement with the 2020 census that looks like model error but is real change.
