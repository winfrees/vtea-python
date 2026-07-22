#!/usr/bin/env bash
# Builds the standalone VTEA runtime with PyInstaller.
#
# Usage: scripts/build_standalone.sh [dist-dir]
#
# Requires vtea-core and vtea-napari[standalone] installed (or run this
# script from a venv where they already are - see
# packaging/pyinstaller/README.md). Produces a single-folder distributable
# at <dist-dir>/vtea-napari/ (default dist-dir: dist/).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${1:-"$REPO_ROOT/dist"}"

pyinstaller "$REPO_ROOT/packaging/pyinstaller/vtea-napari.spec" \
  --distpath "$DIST_DIR" \
  --workpath "$REPO_ROOT/build" \
  --noconfirm

echo "Built: $DIST_DIR/vtea-napari/"
