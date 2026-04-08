#!/usr/bin/env bash
# Run zero-shot search on a target dataset using a pre-trained T-CLSC.
#
# Defaults are tuned for the post-Bug-#003a forecasting setup with ETTh1
# as the target.  Override any variable from the environment, e.g.:
#
#   TARGET_DATASET=ETTh2 ./scripts/zero_shot_search.sh
#   PRETRAIN_ITERS=600 ./scripts/zero_shot_search.sh
#   PRETRAIN_EPOCHS=40 TARGET_DATASET=Epilepsy ./scripts/zero_shot_search.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/datasets}"
COMPARATOR_PATH="${COMPARATOR_PATH:-${ROOT_DIR}/outputs/comparator.pt}"
TARGET_DATASET="${TARGET_DATASET:-ETTh1}"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/default.yaml}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/outputs/best_${TARGET_DATASET}.json}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-}"
# When PRETRAIN_ITERS > 0 it overrides PRETRAIN_EPOCHS for the top-K full-train
# budget on the target.  Leave both empty to use dataset_budgets[target] from
# configs/default.yaml (forecasting → iters=600, others → epochs=40).
PRETRAIN_ITERS="${PRETRAIN_ITERS:-}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-}"

EXTRA_ARGS=""
[ -n "$DEVICE" ]          && EXTRA_ARGS="$EXTRA_ARGS --device $DEVICE"
[ -n "$PRETRAIN_ITERS" ]  && EXTRA_ARGS="$EXTRA_ARGS --pretrain_iters $PRETRAIN_ITERS"
[ -n "$PRETRAIN_EPOCHS" ] && EXTRA_ARGS="$EXTRA_ARGS --pretrain_epochs $PRETRAIN_EPOCHS"

echo "=== ZeroAutoCL: Zero-shot Search ==="
echo "  Comparator : $COMPARATOR_PATH"
echo "  Data dir   : $DATA_DIR"
echo "  Target     : $TARGET_DATASET"
echo "  Config     : $CONFIG"
echo "  Output     : $OUTPUT"
echo "  Iters/Epoch: ${PRETRAIN_ITERS:-from-yaml} / ${PRETRAIN_EPOCHS:-from-yaml}"

python "$SCRIPT_DIR/run_zero_shot_search.py" \
    --comparator_path "$COMPARATOR_PATH" \
    --data_dir        "$DATA_DIR" \
    --target_dataset  "$TARGET_DATASET" \
    --config          "$CONFIG" \
    --output          "$OUTPUT" \
    --seed            "$SEED" \
    $EXTRA_ARGS

echo "=== Done: best config saved to $OUTPUT ==="
