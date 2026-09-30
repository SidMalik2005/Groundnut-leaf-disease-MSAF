#!/usr/bin/env bash
# Downloads the trained MSAF-DenseNet201 weights (msaf_best.pt, 163 MB)
# from the latest GitHub Release into the repository root.
set -euo pipefail

REPO="${REPO:-SidMalik2005/Groundnut-leaf-disease-MSAF}"
ASSET="msaf_best.pt"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/$ASSET}"

if [ -f "$DEST" ]; then
  echo "$DEST already present, skipping."
  exit 0
fi

URL="https://github.com/${REPO}/releases/latest/download/${ASSET}"
echo "Downloading $ASSET from $URL"
curl -L --fail --progress-bar -o "$DEST" "$URL"
echo "Saved to $DEST ($(du -h "$DEST" | cut -f1))"
