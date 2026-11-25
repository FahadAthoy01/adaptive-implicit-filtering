#!/usr/bin/env bash
set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: ./scripts/run_eval.sh <pred_mesh> <gt_mesh> [samples=100000]"
    exit 1
fi

PRED_MESH="$1"
GT_MESH="$2"
SAMPLES="${3:-100000}"

python -m src.eval.metrics --pred "$PRED_MESH" --gt "$GT_MESH" --samples "$SAMPLES"
