#!/bin/bash
# Fetch the four MNIST IDX files into a cache directory.
#
# Run this where there IS network access (a laptop, a login node) and ship the
# directory to wherever training happens -- compute nodes are not assumed to
# reach the internet. ~11.6 MB total.
set -euo pipefail
DEST="${1:-$(dirname "$0")/../data_cache/mnist}"
mkdir -p "$DEST"
MIRRORS=(
  "https://ossci-datasets.s3.amazonaws.com/mnist"
  "https://storage.googleapis.com/cvdf-datasets/mnist"
)
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  out="$DEST/$f.gz"
  [ -s "$out" ] && { echo "have $f.gz"; continue; }
  for m in "${MIRRORS[@]}"; do
    if curl -sSL --max-time 180 -o "$out" "$m/$f.gz"; then echo "got $f.gz from $m"; break; fi
  done
  [ -s "$out" ] || { echo "FAILED to fetch $f.gz" >&2; exit 1; }
done
echo "MNIST cached in $DEST"
