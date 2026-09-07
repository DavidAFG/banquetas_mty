#!/usr/bin/env bash
# Downloads the INEGI inputs for stage 1. Run from Terminal on your Mac:
#     bash fetch_inegi.sh
# Resumable: if it dies, run it again and it picks up where it stopped.
set -euo pipefail
cd "$(dirname "$0")"
RAW="data/raw/monterrey"
mkdir -p "$RAW"

MG_URL="https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/geografia/marcogeo/889463807469/889463807469_s.zip"
MG_ZIP="$RAW/mg2020_nacional.zip"

AGEB_URL="https://www.inegi.org.mx/contenidos/programas/ccpv/2020/datosabiertos/ageb_manzana/ageb_mza_urbana_19_cpv2020_csv.zip"
AGEB_ZIP="$RAW/ageb_mza_urbana_19.zip"

echo "==> 1/2 Censo 2020 por AGEB y manzana urbana, Nuevo Leon (about 8 MB)"
curl -L -C - --retry 5 --retry-delay 3 -o "$AGEB_ZIP" "$AGEB_URL"
unzip -o -q "$AGEB_ZIP" -d "$RAW/censo_ageb_mza"
echo "    extracted to $RAW/censo_ageb_mza"

echo
echo "==> 2/2 Marco Geoestadistico 2020, national (2.89 GB, this is the slow one)"
echo "    Only the Nuevo Leon layers are kept. Resumable if interrupted."
curl -L -C - --retry 5 --retry-delay 3 -o "$MG_ZIP" "$MG_URL"

echo
echo "==> extracting Nuevo Leon (state 19) layers"
mkdir -p "$RAW/mg2020"
unzip -l "$MG_ZIP" | awk '{print $4}' | grep -E '/19[a-z]*\.(shp|dbf|shx|prj|cpg)$' > /tmp/mg19.txt || true
if [ ! -s /tmp/mg19.txt ]; then
  echo "    naming differs from expected, extracting anything with 19 in the name"
  unzip -o -q -j "$MG_ZIP" '*19*' -d "$RAW/mg2020" || true
else
  unzip -o -q -j "$MG_ZIP" $(tr '\n' ' ' < /tmp/mg19.txt) -d "$RAW/mg2020"
fi
# INEGI ships the national bundle as one zip per state, so unpack the inner one
for inner in "$RAW"/mg2020/*.zip; do
  [ -e "$inner" ] || continue
  echo "    unpacking $(basename "$inner")"
  unzip -o -q -j "$inner" 'conjunto_de_datos/*' -d "$RAW/mg2020"
done
ls -la "$RAW/mg2020"/*.shp | head -30

echo
echo "==> done. You can delete $MG_ZIP once the layers above look right."
