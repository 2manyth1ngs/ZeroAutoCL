#!/usr/bin/env bash
# Noisy/clean seed calibration test for ZeroAutoCL.
#
# Quick smoke test (≤ 10 min on RTX 4090):
#   N_STRATEGIES=4 N_SEEDS=1 MAX_EPOCHS=8 ./scripts/calibration.sh
#
# Stage A+B with statistical power (~hours; bumps K to 3 for CV):
#   N_STRATEGIES=32 N_SEEDS=3 MAX_EPOCHS=40 ./scripts/calibration.sh
#
# Re-run analysis only (after tuning gap_threshold or budgets):
#   ANALYZE_ONLY=1 GAP_THRESHOLD=0.05 ./scripts/calibration.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/datasets}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/outputs/calibration_etth2}"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/default.yaml}"
SOURCE_DATASET="${SOURCE_DATASET:-ETTh2}"

N_STRATEGIES="${N_STRATEGIES:-16}"
N_SEEDS="${N_SEEDS:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CROP_LEN="${CROP_LEN:-1024}"
SEED="${SEED:-42}"
GAP_THRESHOLD="${GAP_THRESHOLD:-0.02}"

EXTRA_ARGS=""
[ -n "${DEVICE:-}" ]        && EXTRA_ARGS="$EXTRA_ARGS --device $DEVICE"
[ -n "${BUDGETS:-}" ]       && EXTRA_ARGS="$EXTRA_ARGS --budgets $BUDGETS"
[ -n "${HORIZONS:-}" ]      && EXTRA_ARGS="$EXTRA_ARGS --horizons $HORIZONS"
[ "${ANALYZE_ONLY:-0}" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --analyze_only"

echo "=== ZeroAutoCL: Calibration test ==="
echo "  Source     : $SOURCE_DATASET"
echo "  N×K        : $N_STRATEGIES strategies × $N_SEEDS seeds"
echo "  Max epochs : $MAX_EPOCHS"
echo "  Save dir   : $SAVE_DIR"
echo "  Gap τ      : $GAP_THRESHOLD"

python "$SCRIPT_DIR/run_calibration.py" \
    --data_dir         "$DATA_DIR" \
    --save_dir         "$SAVE_DIR" \
    --config           "$CONFIG" \
    --source_dataset   "$SOURCE_DATASET" \
    --n_strategies     "$N_STRATEGIES" \
    --n_seeds          "$N_SEEDS" \
    --max_epochs       "$MAX_EPOCHS" \
    --batch_size       "$BATCH_SIZE" \
    --crop_len         "$CROP_LEN" \
    --seed             "$SEED" \
    --gap_threshold    "$GAP_THRESHOLD" \
    $EXTRA_ARGS

echo "=== Done: see $SAVE_DIR/calibration_report.md ==="
