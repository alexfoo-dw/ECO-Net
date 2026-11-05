#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATA_DIR="${DATA_DIR:-data}"
CHECKPOINT="${CHECKPOINT:-weights/ECONet-absscene-sample-trained.pth}"
LIMIT=5

# Regular: foreground metrics and combined comparison PNGs.
python eval.py \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA_DIR/abstract_regular.h5" \
  --split test \
  --limit "$LIMIT" \
  --out results/regular_5 \
  --save-pngs \
  "$@"

# Occlusion: full-object metrics and individual full-mask comparison PNGs.
python eval.py \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA_DIR/abstract_regular_occlu.h5" \
  --split test \
  --limit "$LIMIT" \
  --occlusion \
  --out results/occlusion_5 \
  --save-pngs \
  "$@"

# Complex backgrounds: foreground metrics and combined comparison PNGs.
python eval.py \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA_DIR/abstract_regular_complex.h5" \
  --split test \
  --limit "$LIMIT" \
  --out results/complex_5 \
  --save-pngs \
  "$@"