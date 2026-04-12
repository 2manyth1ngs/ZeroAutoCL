#!/usr/bin/env bash
#SBATCH --job-name=zac_full_etth1
#SBATCH --partition=gpu_spot
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --gpu-bind=closest
#SBATCH --exclusive
#SBATCH --constraint=vram24g&cc8.9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/zac_full_etth1_%j.out
#SBATCH --error=logs/zac_full_etth1_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# ZeroAutoCL — Full Plan B pipeline on ETTh1 (CLAUDE_ADV.md §3.7).
#
# Five stages, written to outputs/full_etth1_two_stage/:
#
#   Stage A   : encoder grid search        (search/encoder_grid_search.py)
#               → encoder_grid.json    (36 encoders × 8 sources, GGS fixed)
#   Stage B.1 : seed generation            (search/seed_generator.py)
#               → seeds.json           (top-K_enc × n_per_dataset × 8 sources,
#                                       random CL strategies)
#   Stage B.2 : T-CLSC pretraining         (search/pretrain_comparator.py)
#               → comparator.pt        (curriculum-learned pairwise ranker)
#   Stage B.3 : zero-shot ranking on ETTh1 (search/zero_shot_search.rank_finalists)
#               → finalists.json       (K_enc × top_k_strategies records)
#   Phase 4   : Cartesian retrain           (scripts/run_phase4_finalists.py)
#               → phase4_results.json + phase4_best.json
#
# Hyperparameters live in configs/default.yaml::two_stage_search.
# Defaults (verified on a 4090 24 GB by the §10 protocol A/B test):
#   crop_len      = 1024   (seed gen, configs/default.yaml::seed_generation.crop_len)
#                   3000   (Phase 4 retrain, data/dataset.py::_FORECAST_WINDOW_LEN)
#   batch_size    = 32     (seed_generation.batch_size)
#   pretrain_iters= 600    (dataset_budgets.* and finalists.pretrain_iters)
#   K_enc         = 3      (two_stage_search.encoder_grid.top_k_enc)
#   top_k_strategies = 5   → 3×5 = 15 finalists
#
# Resume mode: each stage skips itself if its output file already exists.
# Delete the file (or the whole directory) to force a re-run.
#
# Submit  :  sbatch scripts/run_full_etth1_two_stage.sbatch
# Watch   :  tail -f logs/zac_full_etth1_<jobid>.out
# Cancel  :  scancel <jobid>
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/ZeroAutoCL}"
cd "$PROJECT_ROOT"

OUT_DIR="outputs/full_etth1_two_stage"
DATA_DIR="data/datasets"
CONFIG="configs/default.yaml"
TARGET="ETTh1"
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

# Source datasets for Stage A and Stage B.1 (target ETTh1 is excluded).
SOURCES=(PEMS03 PEMS04 PEMS07 PEMS08 ExchangeRate ETTm1)

# ─────────────────────────────────────────────────────────────────────────────
# Stage A — encoder grid search (36 encoders × 8 sources, GGS fixed)
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
# Stage B.1 — seed generation (encoder space restricted to top-K_enc)
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Stage B.1: seed generation ═══════"
if [[ -s "${SEEDS_JSON}" ]]; then
    echo "[stage-B.1] ${SEEDS_JSON} exists — skipping (delete to re-run)"
else
    # n_per_dataset and top_k_enc come from configs/default.yaml.
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
# Stage B.3 — zero-shot ranking on ETTh1 → finalists.json
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
# Phase 4 — Cartesian retrain (K_enc × top_k_strategies finalists on ETTh1)
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Phase 4: Cartesian retrain on ${TARGET} ═══════"
if [[ -s "${PHASE4_BEST}" ]]; then
    echo "[phase4] ${PHASE4_BEST} exists — skipping (delete to re-run)"
else
    srun --unbuffered python -u scripts/run_phase4_finalists.py \
        --finalists "${FINALISTS}" \
        --target    "${TARGET}" \
        --data_dir  "${DATA_DIR}" \
        --config    "${CONFIG}" \
        --save_dir  "${OUT_DIR}" \
        --seed      "${SEED}" \
        > >(tee -a "${OUT_DIR}/run.log") 2> >(tee -a "${OUT_DIR}/run.log" >&2)
fi

echo "================================================================="
echo "Finished       : $(date -Is)"
echo "Outputs:"
echo "  Stage A    : ${ENCODER_GRID}"
echo "  Stage B.1  : ${SEEDS_JSON}"
echo "  Stage B.2  : ${COMPARATOR}"
echo "  Stage B.3  : ${FINALISTS}"
echo "  Phase 4    : ${OUT_DIR}/phase4_results.json"
echo "  Best       : ${PHASE4_BEST}"
echo "  Full log   : ${OUT_DIR}/run.log"
echo "================================================================="
