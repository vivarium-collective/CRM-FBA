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
# IMPORTANT: build from the workspace's OWN .venv (crm_dfba + vivarium-workbench
# only). Composite discovery walks every installed process-bigraph package, so
# a shared venv that also has sibling workspaces (viva-munk, spatio-flux, ...)
# leaks their composites into this snapshot. Set one up once with:
#
#     uv venv .venv && VIRTUAL_ENV=.venv uv pip install -e . \
#         && VIRTUAL_ENV=.venv uv pip install vivarium-workbench
#
set -euo pipefail
WS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$WS_ROOT/docs/workbench"
BASE_PATH="/CRM-FBA/workbench"

# Prefer the workspace's own venv binary; fall back to PATH.
PUBLISH="$WS_ROOT/.venv/bin/vivarium-workbench-publish"
[ -x "$PUBLISH" ] || PUBLISH="vivarium-workbench-publish"

rm -rf "$OUT"
# Clear any stale registry / composite-state cache from a previous build.
rm -rf "$WS_ROOT/.pbg/registry-catalog" "$WS_ROOT/.pbg/composite-state-cache"
PYTHONPATH="$WS_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PUBLISH" \
    --workspace "$WS_ROOT" \
    --out "$OUT" \
    --base-path "$BASE_PATH"

# Jekyll (the legacy Pages builder) drops underscore-prefixed files such as
# api/inputs/_global.json; docs/.nojekyll disables Jekyll for the whole site.
find "$OUT" -name '*.map' -delete
touch "$WS_ROOT/docs/.nojekyll"
echo "built read-only workbench at $OUT ($(du -sh "$OUT" | cut -f1))"
