#!/usr/bin/env bash
# Push the evidence frames in batches.
#
# A single push of three thousand files is one long HTTP transfer that fails as
# a unit: if it drops at 90 percent you start again. Committing and pushing in
# batches makes each transfer small and, more importantly, makes the whole
# thing resumable. Run it again after a failure and it picks up exactly where
# it stopped, because it only ever looks at files git does not yet track.
#
#   bash scripts/push_evidence.sh            # batches of 100
#   bash scripts/push_evidence.sh 250        # bigger batches, fewer round trips
set -euo pipefail

BATCH="${1:-100}"
DIR="${2:-site/evidence}"
cd "$(dirname "$0")/.."

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "Not a git repository. Run 'git init' and add a remote first."; exit 1; }
git remote get-url origin >/dev/null 2>&1 || {
  echo "No 'origin' remote. Add one first:"
  echo "  git remote add origin git@github.com:<user>/<repo>.git"; exit 1; }
[ -d "$DIR" ] || { echo "$DIR does not exist. Run build_site_evidence.py first."; exit 1; }

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
remaining() { git ls-files --others --exclude-standard "$DIR" | wc -l | tr -d ' '; }

TOTAL="$(remaining)"
echo "==> $TOTAL untracked files in $DIR, pushing $BATCH at a time to $BRANCH"
[ "$TOTAL" -eq 0 ] && { echo "    nothing to do"; exit 0; }

N=0
while :; do
  mapfile -t FILES < <(git ls-files --others --exclude-standard "$DIR" | head -n "$BATCH")
  [ "${#FILES[@]}" -eq 0 ] && break
  N=$((N+1))
  git add -- "${FILES[@]}"
  git commit -q -m "evidence frames, batch $N (${#FILES[@]} files)"
  # -u on the first push in case the branch has no upstream yet
  if git rev-parse --abbrev-ref "@{upstream}" >/dev/null 2>&1; then
    git push -q origin "$BRANCH"
  else
    git push -q -u origin "$BRANCH"
  fi
  echo "    batch $N pushed, $(remaining) left"
done
echo "==> done, $N batches"
