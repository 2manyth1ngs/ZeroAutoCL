#!/usr/bin/env bash
# Paper-level forecasting experiment: ETTh1 target, all non-ETTh sources.
# Runs phase 1 → 2 → 3 sequentially. Each phase logs to outputs/paper_etth1/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="${ROOT_DIR}/outputs/paper_etth1"
mkdir -p "$OUT_DIR"

# ── Shared paths so all 3 phases agree ─────────────────────────────────────
export DATA_DIR="${ROOT_DIR}/data/datasets"
export CONFIG="${ROOT_DIR}/configs/default.yaml"
export SAVE_DIR="${OUT_DIR}/seeds"
export SEEDS_PATH="${SAVE_DIR}/seeds.json"
export SAVE_PATH="${OUT_DIR}/comparator.pt"
export COMPARATOR_PATH="${SAVE_PATH}"
export TARGET_DATASET="ETTh1"
export OUTPUT="${OUT_DIR}/best_ETTh1.json"
# Source pool: every non-ETTh-family forecasting dataset we have on disk.
export DATASETS="ETTm1 PEMS03 PEMS04 PEMS07 PEMS08 ExchangeRate PEMS-BAY"
export N_PER_DATASET="150"
export BATCH_SIZE="32"
export DEVICE="cuda"
export SEED="42"

START_TS=$(date +%s)
echo "===================================================================="
echo "ZeroAutoCL paper-level forecasting experiment"
echo "  target  : $TARGET_DATASET"
echo "  sources : $DATASETS"
echo "  outdir  : $OUT_DIR"
echo "  started : $(date)"
echo "===================================================================="

echo ""
echo ">>> Phase 1: generate seeds"
"${SCRIPT_DIR}/generate_seeds.sh" 2>&1 | tee "${OUT_DIR}/phase1_seeds.log"
P1_TS=$(date +%s)
echo ">>> Phase 1 done in $(( (P1_TS - START_TS) / 60 )) min"

echo ""
echo ">>> Phase 2: pretrain comparator"
"${SCRIPT_DIR}/pretrain_comparator.sh" 2>&1 | tee "${OUT_DIR}/phase2_comparator.log"
P2_TS=$(date +%s)
echo ">>> Phase 2 done in $(( (P2_TS - P1_TS) / 60 )) min"

echo ""
echo ">>> Phase 3: zero-shot search on $TARGET_DATASET"
"${SCRIPT_DIR}/zero_shot_search.sh" 2>&1 | tee "${OUT_DIR}/phase3_search.log"
P3_TS=$(date +%s)
echo ">>> Phase 3 done in $(( (P3_TS - P2_TS) / 60 )) min"

echo ""
echo "===================================================================="
echo "All phases complete in $(( (P3_TS - START_TS) / 60 )) min total"
echo "Best config: ${OUTPUT}"
echo "===================================================================="
