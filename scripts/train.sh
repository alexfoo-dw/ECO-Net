#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATA_DIR="${DATA_DIR:-data}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/abstract_regular.h5}"
CONFIG="${CONFIG:-configs/absscene.json}"
OUT_DIR="${OUT_DIR:-runs/absscene}"
DEVICE="${DEVICE:-cuda:0}"
MODEL_NAME="${MODEL_NAME:-ECONet-absscene-sample-trained.pth}"

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "Training dataset not found: $TRAIN_DATA" >&2
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Configuration not found: $CONFIG" >&2
    exit 1
fi

python train.py \
    --config "$CONFIG" \
    --data "$TRAIN_DATA" \
    --split train \
    --out "$OUT_DIR" \
    --device "$DEVICE" \
    --model-name "$MODEL_NAME" \
    "$@"