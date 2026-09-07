#!/usr/bin/env bash
# Stage 0: coverage audit. Run this from Terminal on your Mac, from the repo folder:
#     bash run_stage0.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null; then
  echo "python3 not found. Install it (brew install python) and run again."; exit 1
fi

if [ ! -d .venv ]; then
  echo "==> creating virtual environment"
  python3 -m venv .venv
fi
source .venv/bin/activate

echo "==> installing dependencies (first run only, a few minutes)"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

if [ ! -f .env ]; then echo ".env missing"; exit 1; fi
source .env

mkdir -p logs
echo "==> running stage 0, logging to logs/stage0.log"
python -m banquetas coverage --city monterrey 2>&1 | tee logs/stage0.log
echo
echo "==> done. Outputs in data/out/monterrey/"
