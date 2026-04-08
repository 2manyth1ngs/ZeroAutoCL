#!/usr/bin/env bash
# Generate seed data for T-CLSC pre-training.
# Defaults are tuned for the post-Bug-#003a forecasting setup with ETTh1
# as the target.  Override any variable from the environment, e.g.:
#
#   DATASETS="HAR Epilepsy" PRETRAIN_EPOCHS=40 ./scripts/generate_seeds.sh
#   PRETRAIN_ITERS=600 N_PER_DATASET=50 ./scripts/generate_seeds.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/datasets}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/outputs/seeds}"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/default.yaml}"
# Default sources: every forecasting dataset except ETTh1 (the target).
# Per CLAUDE.md spec, ETTh2 is a SOURCE — only ETTh1 is held out for evaluation.
DATASETS="${DATASETS:-ETTh2 ETTm1 PEMS03 PEMS04 PEMS07 PEMS08 ExchangeRate PEMS-BAY}"
N_PER_DATASET="${N_PER_DATASET:-30}"
# When PRETRAIN_ITERS > 0 it overrides PRETRAIN_EPOCHS *for every dataset*.
# Leave both empty to use the per-dataset budget from configs/default.yaml
# (forecasting → iters=600, classification/anomaly → epochs=40).
PRETRAIN_ITERS="${PRETRAIN_ITERS:-}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-}"

EXTRA_ARGS=""
[ -n "$DEVICE" ]          && EXTRA_ARGS="$EXTRA_ARGS --device $DEVICE"
[ -n "$PRETRAIN_ITERS" ]  && EXTRA_ARGS="$EXTRA_ARGS --pretrain_iters $PRETRAIN_ITERS"
[ -n "$PRETRAIN_EPOCHS" ] && EXTRA_ARGS="$EXTRA_ARGS --pretrain_epochs $PRETRAIN_EPOCHS"

echo "=== ZeroAutoCL: Generate Seeds ==="
echo "  Data dir   : $DATA_DIR"
echo "  Save dir   : $SAVE_DIR"
echo "  Datasets   : $DATASETS"
echo "  Config     : $CONFIG"
echo "  N_PER      : $N_PER_DATASET"
echo "  Iters/Epoch: ${PRETRAIN_ITERS:-from-yaml} / ${PRETRAIN_EPOCHS:-from-yaml}"

python "$SCRIPT_DIR/run_generate_seeds.py" \
    --data_dir         "$DATA_DIR" \
    --save_dir         "$SAVE_DIR" \
    --config           "$CONFIG" \
    --datasets         $DATASETS \
    --n_per_dataset    "$N_PER_DATASET" \
    --batch_size       "$BATCH_SIZE" \
    --seed             "$SEED" \
    $EXTRA_ARGS

echo "=== Done: seeds saved to $SAVE_DIR/seeds.json ==="
