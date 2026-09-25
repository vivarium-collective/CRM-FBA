#!/usr/bin/env bash
# Rebuild the read-only vivarium-workbench snapshot into docs/workbench/.
#
# GitHub Pages for this repo serves from main:/docs (legacy source), so the
# read-only workbench is committed under docs/workbench/ alongside the
# experiment report at docs/index.html — no gh-pages branch involved. Once
# merged to main it is live at:
#
#     https://vivarium-collective.github.io/CRM-FBA/workbench/
#
# Requires `vivarium-workbench-publish` on PATH (from a venv with
# vivarium-workbench installed). Run from the workspace root.
set -euo pipefail
WS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$WS_ROOT/docs/workbench"
BASE_PATH="/CRM-FBA/workbench"

rm -rf "$OUT"
PYTHONPATH="$WS_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  vivarium-workbench-publish \
    --workspace "$WS_ROOT" \
    --out "$OUT" \
    --base-path "$BASE_PATH"

# Jekyll (the legacy Pages builder) drops underscore-prefixed files such as
# api/inputs/_global.json; docs/.nojekyll disables Jekyll for the whole site.
find "$OUT" -name '*.map' -delete
touch "$WS_ROOT/docs/.nojekyll"
echo "built read-only workbench at $OUT ($(du -sh "$OUT" | cut -f1))"
