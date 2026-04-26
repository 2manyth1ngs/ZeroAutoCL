#!/usr/bin/env bash
#SBATCH --job-name=zac_autocts_etth1
#SBATCH --partition=gpu_spot
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --gpu-bind=closest
#SBATCH --exclusive
#SBATCH --constraint=vram24g&cc8.9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --output=logs/zac_autocts_etth1_%j.out
#SBATCH --error=logs/zac_autocts_etth1_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# ZeroAutoCL — AutoCTS++-aligned two-stage pipeline on ETTh1.
#
# Differences vs scripts/run_full_etth1_two_stage.sbatch
# -----------------------------------------------------
# That older script runs "Plan B" (encoder_grid + single-mode seed gen).
# This script runs the AutoCTS++ reference protocol we implemented in
# P0 + P1 + P2:
#
#   Stage A    : encoder grid search (unchanged)
#                → encoder_grid.json
#   Stage B.1a : NOISY seeds, deterministic init      (--mode noisy)
#                → seeds_noisy_fixed/seeds.json
#   Stage B.1b : NOISY seeds, random init             (--mode noisy --randomise_init)
#                → seeds_noisy_random/seeds.json
#   Stage B.1c : CLEAN seeds, full training           (--mode clean)
#                → seeds_clean/seeds.json
#   Stage B.2  : T-CLSC two-stage training (noisy pretrain → clean finetune)
#                → comparator.pt
#   Stage B.3  : zero-shot ranking on ETTh1
#                → finalists.json
#   Phase 4    : Cartesian retrain (K_enc × top_k_strategies) on ETTh1
#                → phase4_results.json + phase4_best.json
#
# Per-task seed budget: 20 noisy-fixed + 20 noisy-random + 20 clean = 60.
# This matches AutoCTS++ ``sample_num=20`` (reference/AutoCTS_plusplus/
# exps/generate_seeds.py L94-103).  Across 6 source tasks → 360 records.
#
# Mode-specific iter budget (CLAUDE_ADV §12 calibration, 2026-04-25):
# YAML ``dataset_budgets[*].noisy_pretrain_iters: 200`` cuts the noisy
# stages (B.1a + B.1b) to 1/3 of the clean budget without crossing the
# calibrated ranking-quality cliff (Spearman ρ=0.905 / pairwise
# concordance=0.947 already at ≈1 epoch ≈ 240 iters on ETTh2).
# Override via ``--noisy_pretrain_iters N`` if you re-calibrate per-dataset.
#
# Wall-clock estimate (RTX 4090, 24 GB):
#   Stage A     : 27 encoders × 3 sources × ~1.6 min ≈  2.2 h *
#   Stage B.1a  : 20 × 6 × ~1.7 min ≈ 3.4 h         **(was 10 h @ 600 iters)**
#   Stage B.1b  : 20 × 6 × ~1.7 min ≈ 3.4 h         **(was 10 h @ 600 iters)**
#   Stage B.1c  : 20 × 6 × ~6   min ≈ 12 h
#   Stage B.2+3 : ≤ 1 h
#   Phase 4     : 15 finalists × ~45 min ≈ 11 h
#                                          -------
#                                          ≈ 33 h  (was 46 h before §12 fix)
#
# * Stage A speedups (vs. previous 11 h estimate):
#     - n_layers=4 dropped from ENCODER_CHOICES   (36 → 27 encoders, -25%)
#     - STAGE_A_SOURCES reduced from 6 → 3        (representative split, -50%)
#     - YAML two_stage_search.encoder_grid.pretrain_iters: 300  (-50% train)
#     - YAML two_stage_search.encoder_grid.eval_horizons: [24,168,336]
#       drops H=720 + H=48                        (cheaper Ridge fits)
#   Seed gen (B.1c clean) and Phase 4 still use full 6-source set + 600-iter
#   budget; only the noisy stages (B.1a/b) consume the 200-iter knob.
#
# Resume mode: every stage skips itself if its output file exists.  Delete
# the specific file (or the whole ``${OUT_DIR}``) to force a re-run.
#
# Submit  :  sbatch scripts/run_full_etth1_autocts.sbatch
# Watch   :  tail -f logs/zac_autocts_etth1_<jobid>.out
# Cancel  :  scancel <jobid>
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/ZeroAutoCL}"
cd "$PROJECT_ROOT"

OUT_DIR="outputs/full_etth1_autocts"
DATA_DIR="data/datasets"
CONFIG="configs/default.yaml"
TARGET="ETTh1"

# Separate RNG seeds for each seed-generation sub-run so the two noisy
# halves and the clean run sample DIFFERENT strategies.
SEED=42
SEED_NOISY_FIXED=42
SEED_NOISY_RANDOM=43
SEED_CLEAN=44

ENCODER_GRID="${OUT_DIR}/encoder_grid.json"
SEEDS_NOISY_FIXED_DIR="${OUT_DIR}/seeds_noisy_fixed"
SEEDS_NOISY_RANDOM_DIR="${OUT_DIR}/seeds_noisy_random"
SEEDS_CLEAN_DIR="${OUT_DIR}/seeds_clean"
SEEDS_NOISY_FIXED="${SEEDS_NOISY_FIXED_DIR}/seeds.json"
SEEDS_NOISY_RANDOM="${SEEDS_NOISY_RANDOM_DIR}/seeds.json"
SEEDS_CLEAN="${SEEDS_CLEAN_DIR}/seeds.json"
COMPARATOR="${OUT_DIR}/comparator.pt"
FINALISTS="${OUT_DIR}/finalists.json"
PHASE4_BEST="${OUT_DIR}/phase4_best.json"

mkdir -p logs "${OUT_DIR}" "${SEEDS_NOISY_FIXED_DIR}" \
         "${SEEDS_NOISY_RANDOM_DIR}" "${SEEDS_CLEAN_DIR}"

echo "================================================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Node           : $(hostname)"
echo "Partition      : ${SLURM_JOB_PARTITION}"
echo "GPUs visible   : ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Project root   : ${PROJECT_ROOT}"
echo "Output dir     : ${OUT_DIR}"
echo "Target         : ${TARGET} (forecasting, AutoCTS++-aligned 2-stage)"
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

# Source datasets for Stage B.1 (target ETTh1 is excluded — strict zero-shot).
# PEMS07 leads the list — it's the heaviest source (883 sensor channels)
# and the most likely to OOM, so failing fast here saves us the wall-clock
# of all the other sources before discovering a crash.  Per-dataset
# checkpoints (`_seeds_<ds>.json` in each save_dir) make completed sources
# resume-safe regardless of order.
SOURCES=(PEMS07 ETTh2 PEMS03 PEMS04 PEMS08 ExchangeRate)

# Stage A only needs a coarse encoder ranking, so we use a 3-source
# representative subset instead of all 6:
#   PEMS04        → wide multivariate (307 vars, traffic)
#   ExchangeRate  → narrow multivariate (8 vars, finance)
#   ETTh2         → ETT-style (7 vars, energy; closest to target ETTh1)
# These three span the channel-count and domain-distribution diversity of
# the full source pool. Seed generation (B.1*) keeps the full 6-source set.
STAGE_A_SOURCES=(PEMS04 ExchangeRate ETTh2)

# Read CLI-driven scalars from YAML once.
N_PER_DATASET=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}', encoding='utf-8')); print(c['two_stage_search']['cl_strategy_search']['n_per_dataset'])")
TOP_K_STR=$(python -c    "import yaml; c=yaml.safe_load(open('${CONFIG}', encoding='utf-8')); print(c['two_stage_search']['cl_strategy_search']['top_k_strategies'])")
# Stage A is skipped; Stage B uses the GGS encoder only (n_layers=10, hidden=64,
# output=320 — encoder_default in configs/default.yaml).  Hard-set to 1 so
# downstream stages do not multiply over phantom encoders.
TOP_K_ENC=1
echo "YAML scalars   : n_per_dataset=${N_PER_DATASET}  top_k_enc=${TOP_K_ENC} (forced, GGS only)  top_k_strategies=${TOP_K_STR}"

# ─────────────────────────────────────────────────────────────────────────────
# Stage A — SKIPPED.  Empirically the grid's top encoders cluster around the
# GGS default (n_layers=10, output_dim=320).  We bypass the 2-3 h grid search
# and write a synthetic encoder_grid.json containing only the GGS config so
# the rest of the pipeline (which expects this file) still works unchanged.
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Stage A: SKIPPED — using GGS encoder ═══════"
if [[ -s "${ENCODER_GRID}" ]]; then
    echo "[stage-A] ${ENCODER_GRID} exists — skipping (delete to re-run)"
else
    python - <<EOF
import json, yaml
with open("${CONFIG}", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
ggs = cfg.get("encoder_default", {"n_layers": 10, "hidden_dim": 64, "output_dim": 320})
records = [{
    "encoder_config": {
        "n_layers":   int(ggs["n_layers"]),
        "hidden_dim": int(ggs["hidden_dim"]),
        "output_dim": int(ggs["output_dim"]),
    },
    "mean_perf": 0.0,
    "per_dataset": {},
    "stage_a_skipped": True,
}]
with open("${ENCODER_GRID}", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)
print(f"[stage-A] wrote synthetic encoder_grid.json with GGS-only encoder: {records[0]['encoder_config']}")
EOF
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage B.1a — NOISY seeds, deterministic init
#   AutoCTS++ analog: generate_seeds.py --mode noisy_seeds (use_seed=True branch)
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Stage B.1a: noisy seeds (deterministic init) ═══════"
if [[ -s "${SEEDS_NOISY_FIXED}" ]]; then
    echo "[stage-B.1a] ${SEEDS_NOISY_FIXED} exists — skipping (delete to re-run)"
else
    srun --unbuffered python -u scripts/run_generate_seeds.py \
        --data_dir       "${DATA_DIR}" \
        --save_dir       "${SEEDS_NOISY_FIXED_DIR}" \
        --config         "${CONFIG}" \
        --datasets       "${SOURCES[@]}" \
        --n_per_dataset  "${N_PER_DATASET}" \
        --encoder_grid   "${ENCODER_GRID}" \
        --top_k_enc      "${TOP_K_ENC}" \
        --mode           noisy \
        --seed           "${SEED_NOISY_FIXED}" \
        > >(tee -a "${OUT_DIR}/run.log") 2> >(tee -a "${OUT_DIR}/run.log" >&2)
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage B.1b — NOISY seeds, random init (system entropy per candidate)
#   AutoCTS++ analog: generate_seeds.py --mode noisy_seeds (use_seed=False branch)
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Stage B.1b: noisy seeds (random init) ═══════"
if [[ -s "${SEEDS_NOISY_RANDOM}" ]]; then
    echo "[stage-B.1b] ${SEEDS_NOISY_RANDOM} exists — skipping (delete to re-run)"
else
    srun --unbuffered python -u scripts/run_generate_seeds.py \
        --data_dir       "${DATA_DIR}" \
        --save_dir       "${SEEDS_NOISY_RANDOM_DIR}" \
        --config         "${CONFIG}" \
        --datasets       "${SOURCES[@]}" \
        --n_per_dataset  "${N_PER_DATASET}" \
        --encoder_grid   "${ENCODER_GRID}" \
        --top_k_enc      "${TOP_K_ENC}" \
        --mode           noisy \
        --randomise_init \
        --seed           "${SEED_NOISY_RANDOM}" \
        > >(tee -a "${OUT_DIR}/run.log") 2> >(tee -a "${OUT_DIR}/run.log" >&2)
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage B.1c — CLEAN seeds (full training budget, deterministic init)
#   AutoCTS++ analog: generate_seeds.py --mode clean_seeds
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Stage B.1c: clean seeds ═══════"
if [[ -s "${SEEDS_CLEAN}" ]]; then
    echo "[stage-B.1c] ${SEEDS_CLEAN} exists — skipping (delete to re-run)"
else
    srun --unbuffered python -u scripts/run_generate_seeds.py \
        --data_dir       "${DATA_DIR}" \
        --save_dir       "${SEEDS_CLEAN_DIR}" \
        --config         "${CONFIG}" \
        --datasets       "${SOURCES[@]}" \
        --n_per_dataset  "${N_PER_DATASET}" \
        --encoder_grid   "${ENCODER_GRID}" \
        --top_k_enc      "${TOP_K_ENC}" \
        --mode           clean \
        --seed           "${SEED_CLEAN}" \
        > >(tee -a "${OUT_DIR}/run.log") 2> >(tee -a "${OUT_DIR}/run.log" >&2)
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage B.2 — T-CLSC two-stage comparator training
#   Stage 1 (pretrain) : both noisy seed pools concatenated (symmetric pairs,
#                        gap-split valid, patience=5 early stop)
#   Stage 2 (finetune) : clean seed pool, lr=0.1× pretrain_lr, epochs÷2
#                        starts from best-by-noisy-valid weights
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Stage B.2: T-CLSC two-stage training ═══════"
if [[ -s "${COMPARATOR}" ]]; then
    echo "[stage-B.2] ${COMPARATOR} exists — skipping (delete to re-run)"
else
    srun --unbuffered python -u scripts/run_pretrain_comparator.py \
        --seeds_path       "${SEEDS_NOISY_FIXED}" "${SEEDS_NOISY_RANDOM}" \
        --clean_seeds_path "${SEEDS_CLEAN}" \
        --data_dir         "${DATA_DIR}" \
        --save_path        "${COMPARATOR}" \
        --config           "${CONFIG}" \
        --seed             "${SEED}" \
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
    srun --unbuffered python -u scripts/run_zero_shot_search.py \
        --comparator_path  "${COMPARATOR}" \
        --data_dir         "${DATA_DIR}" \
        --target_dataset   "${TARGET}" \
        --config           "${CONFIG}" \
        --encoder_grid     "${ENCODER_GRID}" \
        --top_k_enc        "${TOP_K_ENC}" \
        --top_k_strategies "${TOP_K_STR}" \
        --output           "${FINALISTS}" \
        --seed             "${SEED}" \
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

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "================================================================="
echo "Finished       : $(date -Is)"
echo "Task type      : forecasting (AutoCTS++-aligned)"
echo "Target         : ${TARGET}"
echo "Sources        : ${SOURCES[*]}"
echo "Seed budget    : ${N_PER_DATASET}/mode × 3 modes × ${#SOURCES[@]} tasks = $(( N_PER_DATASET * 3 * ${#SOURCES[@]} )) records"
echo "Outputs:"
echo "  Stage A      : ${ENCODER_GRID}"
echo "  Stage B.1a   : ${SEEDS_NOISY_FIXED}"
echo "  Stage B.1b   : ${SEEDS_NOISY_RANDOM}"
echo "  Stage B.1c   : ${SEEDS_CLEAN}"
echo "  Stage B.2    : ${COMPARATOR}"
echo "  Stage B.3    : ${FINALISTS}"
echo "  Phase 4      : ${OUT_DIR}/phase4_results.json"
echo "  Best         : ${PHASE4_BEST}"
echo "  Full log     : ${OUT_DIR}/run.log"
echo "================================================================="

if [[ -s "${PHASE4_BEST}" ]]; then
    echo
    echo ">>> Best finalist:"
    python -c "
import json
with open('${PHASE4_BEST}') as f:
    b = json.load(f)
print(f\"  Tag      : {b.get('tag', 'n/a')}\")
for k in ('mean_mse', 'mean_mae', 'mean_mse_raw', 'mean_mae_raw'):
    if k in b:
        print(f\"  {k:9s}: {b[k]:.4f}\")
print(f\"  Encoder  : {b.get('encoder_config', 'n/a')}\")
"
fi
