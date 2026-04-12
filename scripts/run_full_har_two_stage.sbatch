#!/usr/bin/env bash
#SBATCH --job-name=zac_full_har
#SBATCH --partition=gpu_spot
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --gpu-bind=closest
#SBATCH --exclusive
#SBATCH --constraint=vram24g&cc8.9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --output=logs/zac_full_har_%j.out
#SBATCH --error=logs/zac_full_har_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# ZeroAutoCL — Full Plan B pipeline on HAR (classification task).
#
# Five stages, written to outputs/full_har_two_stage/:
#
#   Stage A   : encoder grid search        (scripts/run_encoder_grid.py)
#               → encoder_grid.json    (36 encoders × 4 sources, GGS fixed)
#   Stage B.1 : seed generation            (scripts/run_generate_seeds.py)
#               → seeds.json           (top-K_enc × n_per_dataset × 4 sources,
#                                       random CL strategies)
#   Stage B.2 : T-CLSC pretraining         (search/pretrain_comparator.py)
#               → comparator.pt        (curriculum-learned pairwise ranker)
#   Stage B.3 : zero-shot ranking on HAR   (search/zero_shot_search.rank_finalists)
#               → finalists.json       (K_enc × top_k_strategies records)
#   Phase 4   : Cartesian retrain           (scripts/run_phase4_finalists_cls.py)
#               → phase4_results.json + phase4_best.json
#
# Classification-specific settings vs forecasting:
#   - Epoch-based training (40 epochs, not iter-based)
#   - Val-best checkpointing enabled (val signal reliable for classification)
#   - Evaluation via SVM on pooled embeddings (not Ridge regression)
#   - Metric: accuracy / macro-F1 (not MSE/MAE)
#   - No crop_len windowing (classification samples are naturally discrete)
#
# Source datasets : Epilepsy, SleepEEG, Gesture, NATOPS
# Target dataset  : HAR (excluded from source pool — strict zero-shot)
#
# Resume mode: each stage skips itself if its output file already exists.
# Delete the file (or the whole directory) to force a re-run.
#
# Submit  :  sbatch scripts/run_full_har_two_stage.sbatch
# Watch   :  tail -f logs/zac_full_har_<jobid>.out
# Cancel  :  scancel <jobid>
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/ZeroAutoCL2/ZeroAutoCL}"
cd "$PROJECT_ROOT"

OUT_DIR="outputs/full_har_two_stage"
DATA_DIR="data/datasets"
CONFIG="configs/default.yaml"
TARGET="HAR"
SEED=42

ENCODER_GRID="${OUT_DIR}/encoder_grid.json"
SEEDS_JSON="${OUT_DIR}/seeds.json"
COMPARATOR="${OUT_DIR}/comparator.pt"
FINALISTS="${OUT_DIR}/finalists.json"
PHASE4_BEST="${OUT_DIR}/phase4_best.json"

mkdir -p logs "${OUT_DIR}"

echo "================================================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Node           : $(hostname)"
echo "Partition      : ${SLURM_JOB_PARTITION}"
echo "GPUs visible   : ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Project root   : ${PROJECT_ROOT}"
echo "Output dir     : ${OUT_DIR}"
echo "Target         : ${TARGET} (classification)"
echo "Started        : $(date -Is)"
echo "================================================================="

# ── Software environment ────────────────────────────────────────────────────
module load anaconda
conda activate zeroautocl

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${SEED}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| dev:', torch.cuda.get_device_name(0), '| vram_gb:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))"

# Source datasets for classification (HAR excluded — it's the target).
SOURCES=(Epilepsy SleepEEG Gesture NATOPS)

# ─────────────────────────────────────────────────────────────────────────────
# Stage A — encoder grid search (36 encoders × 4 sources, GGS fixed)
#
# Classification datasets use epoch-based budgets (40 epochs each).
# No crop_len needed — classification samples are naturally discrete.
# Estimated time: 36 × 4 = 144 runs × ~2 min each ≈ 5 hours.
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Stage A: encoder grid search ═══════"
if [[ -s "${ENCODER_GRID}" ]]; then
    echo "[stage-A] ${ENCODER_GRID} exists — skipping (delete to re-run)"
else
    srun --unbuffered python -u scripts/run_encoder_grid.py \
        --data_dir "${DATA_DIR}" \
        --save_dir "${OUT_DIR}" \
        --config   "${CONFIG}" \
        --datasets "${SOURCES[@]}" \
        --seed     "${SEED}" \
        > >(tee -a "${OUT_DIR}/run.log") 2> >(tee -a "${OUT_DIR}/run.log" >&2)
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage B.1 — seed generation (top-K_enc encoders × random CL strategies)
#
# 200 candidates × 4 sources = 800 seed records.
# Each candidate: 40 epoch CL pretrain + SVM eval.
# Estimated time: 800 × ~2 min ≈ 27 hours.
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Stage B.1: seed generation ═══════"
if [[ -s "${SEEDS_JSON}" ]]; then
    echo "[stage-B.1] ${SEEDS_JSON} exists — skipping (delete to re-run)"
else
    N_PER_DATASET=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['two_stage_search']['cl_strategy_search']['n_per_dataset'])")
    TOP_K_ENC=$(python -c    "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['two_stage_search']['encoder_grid']['top_k_enc'])")
    echo "[stage-B.1] n_per_dataset=${N_PER_DATASET}  top_k_enc=${TOP_K_ENC}"

    srun --unbuffered python -u scripts/run_generate_seeds.py \
        --data_dir       "${DATA_DIR}" \
        --save_dir       "${OUT_DIR}" \
        --config         "${CONFIG}" \
        --datasets       "${SOURCES[@]}" \
        --n_per_dataset  "${N_PER_DATASET}" \
        --encoder_grid   "${ENCODER_GRID}" \
        --top_k_enc      "${TOP_K_ENC}" \
        --seed           "${SEED}" \
        > >(tee -a "${OUT_DIR}/run.log") 2> >(tee -a "${OUT_DIR}/run.log" >&2)
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage B.2 — T-CLSC comparator pretraining
#
# Learns pairwise ranking from seed records via curriculum learning.
# Estimated time: < 30 min (comparator is lightweight).
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Stage B.2: T-CLSC pretraining ═══════"
if [[ -s "${COMPARATOR}" ]]; then
    echo "[stage-B.2] ${COMPARATOR} exists — skipping (delete to re-run)"
else
    srun --unbuffered python -u scripts/run_pretrain_comparator.py \
        --seeds_path "${SEEDS_JSON}" \
        --data_dir   "${DATA_DIR}" \
        --save_path  "${COMPARATOR}" \
        --config     "${CONFIG}" \
        --seed       "${SEED}" \
        > >(tee -a "${OUT_DIR}/run.log") 2> >(tee -a "${OUT_DIR}/run.log" >&2)
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage B.3 — zero-shot ranking on HAR → finalists.json
#
# Swiss-system tournament: 200k candidates, 25 rounds.
# Output: K_enc × top_k_strategies finalists (default 3×5 = 15).
# Estimated time: < 10 min (comparator forward only, no training).
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Stage B.3: zero-shot ranking on ${TARGET} ═══════"
if [[ -s "${FINALISTS}" ]]; then
    echo "[stage-B.3] ${FINALISTS} exists — skipping (delete to re-run)"
else
    TOP_K_ENC=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['two_stage_search']['encoder_grid']['top_k_enc'])")
    TOP_K_STR=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['two_stage_search']['cl_strategy_search']['top_k_strategies'])")
    echo "[stage-B.3] top_k_enc=${TOP_K_ENC}  top_k_strategies=${TOP_K_STR}"

    srun --unbuffered python -u scripts/run_zero_shot_search.py \
        --comparator_path "${COMPARATOR}" \
        --data_dir        "${DATA_DIR}" \
        --target_dataset  "${TARGET}" \
        --config          "${CONFIG}" \
        --encoder_grid    "${ENCODER_GRID}" \
        --top_k_enc       "${TOP_K_ENC}" \
        --top_k_strategies "${TOP_K_STR}" \
        --output          "${FINALISTS}" \
        --seed            "${SEED}" \
        > >(tee -a "${OUT_DIR}/run.log") 2> >(tee -a "${OUT_DIR}/run.log" >&2)
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Cartesian retrain (K_enc × top_k_strategies finalists on HAR)
#
# Classification-specific:
#   - 40 epoch pretrain with val-best checkpointing
#   - eval_classification (SVM) on train→test and train→val
#   - Best selected by test accuracy (higher = better)
#
# Estimated time: 15 finalists × ~2 min ≈ 30 min.
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Phase 4: Cartesian retrain on ${TARGET} ═══════"
if [[ -s "${PHASE4_BEST}" ]]; then
    echo "[phase4] ${PHASE4_BEST} exists — skipping (delete to re-run)"
else
    srun --unbuffered python -u scripts/run_phase4_finalists_cls.py \
        --finalists "${FINALISTS}" \
        --target    "${TARGET}" \
        --data_dir  "${DATA_DIR}" \
        --config    "${CONFIG}" \
        --save_dir  "${OUT_DIR}" \
        --seed      "${SEED}" \
        > >(tee -a "${OUT_DIR}/run.log") 2> >(tee -a "${OUT_DIR}/run.log" >&2)
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "================================================================="
echo "Finished       : $(date -Is)"
echo "Task type      : classification"
echo "Target         : ${TARGET}"
echo "Sources        : ${SOURCES[*]}"
echo "Outputs:"
echo "  Stage A    : ${ENCODER_GRID}"
echo "  Stage B.1  : ${SEEDS_JSON}"
echo "  Stage B.2  : ${COMPARATOR}"
echo "  Stage B.3  : ${FINALISTS}"
echo "  Phase 4    : ${OUT_DIR}/phase4_results.json"
echo "  Best       : ${PHASE4_BEST}"
echo "  Full log   : ${OUT_DIR}/run.log"
echo "================================================================="

# Print best result if available.
if [[ -s "${PHASE4_BEST}" ]]; then
    echo
    echo ">>> Best finalist:"
    python -c "
import json
with open('${PHASE4_BEST}') as f:
    b = json.load(f)
print(f\"  Tag      : {b['tag']}\")
print(f\"  Test Acc : {b['test_acc']:.4f}\")
print(f\"  Test F1  : {b['test_f1']:.4f}\")
print(f\"  Val Acc  : {b['val_acc']:.4f}\")
print(f\"  Encoder  : {b['encoder_config']}\")
"
fi
