# Banquetas Monterrey

An interactive audit of sidewalk provision on Monterrey's arterial and collector
network, measured from Kaart's 2020 street level imagery.

`index.html` carries the page, the data and the styling in one file; the basemap and
the MapLibre library load from the network at run time. `evidence/` holds one frame
per block face, the photograph above the segmentation the model read it from, so any
claim on the map can be checked against the pixels it came from.

Generate `evidence/` before deploying:

```
python scripts/build_site_evidence.py monterrey
```

The map degrades gracefully without it: the panel keeps every number and the link to
Mapillary, and says the frames were not deployed with that copy.

## Publishing on GitHub Pages

1. Create a repository and put `index.html` and `evidence/` at its root.
2. Settings, then Pages, then set Source to "Deploy from a branch" and pick `main`
   with folder `/ (root)`.
3. The site appears at `https://<user>.github.io/<repo>/` within a minute or two.

Nothing needs building and there is no Jekyll step to worry about, since the site is
static with no underscore-prefixed paths. The evidence set is the bulk of the
repository; if the push feels heavy, regenerate it at a lower quality.

## Opening it locally

Double click it. It works from `file://` because the data is embedded rather than
fetched, which a local page is not allowed to do.

## What is on the map

Each line is one *frente de manzana*, one side of a street between two corners,
which is the unit INEGI records sidewalk presence on. Colour carries provision and
line width repeats it. A dashed core marks a face read from a single pass.
Motorway and trunk faces are drawn but excluded from every figure, because
pedestrians are prohibited on them.

Presence is validated at 0.81 accuracy and 0.86 F1 per block face, against a 0.695
majority baseline, on 177 hand labelled frames drawn evenly across socioeconomic
deciles. Width is indicative: mean absolute error 0.68 m against 33 faces measured
in Google Earth, so it is reliable in aggregate and not per face.

## Credits

- Imagery © Kaart, CC BY-SA, via Mapillary
- Street network © OpenStreetMap contributors; basemap by CARTO
- Municipal boundaries and socioeconomic variables from INEGI, Marco Geoestadístico
  and Censo de Población y Vivienda 2020
- Segmentation with `facebook/mask2former-swin-large-mapillary-vistas-semantic`
