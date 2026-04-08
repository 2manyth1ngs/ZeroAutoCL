#!/usr/bin/env bash
# Pre-train the T-CLSC comparator with curriculum learning.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/datasets}"
SEEDS_PATH="${SEEDS_PATH:-${ROOT_DIR}/outputs/seeds/seeds.json}"
SAVE_PATH="${SAVE_PATH:-${ROOT_DIR}/outputs/comparator.pt}"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/default.yaml}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-}"

DEVICE_ARG=""
if [ -n "$DEVICE" ]; then
    DEVICE_ARG="--device $DEVICE"
fi

echo "=== ZeroAutoCL: Pre-train Comparator ==="
echo "  Seeds      : $SEEDS_PATH"
echo "  Data dir   : $DATA_DIR"
echo "  Output     : $SAVE_PATH"
echo "  Config     : $CONFIG"

python "$SCRIPT_DIR/run_pretrain_comparator.py" \
    --seeds_path  "$SEEDS_PATH" \
    --data_dir    "$DATA_DIR" \
    --save_path   "$SAVE_PATH" \
    --config      "$CONFIG" \
    --seed        "$SEED" \
    $DEVICE_ARG

echo "=== Done: comparator saved to $SAVE_PATH ==="
