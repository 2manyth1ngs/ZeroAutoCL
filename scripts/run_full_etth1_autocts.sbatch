#!/usr/bin/env bash
#SBATCH --job-name=zac_autocts_etth1
#SBATCH --partition=gpu_spot
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --gpu-bind=closest
#SBATCH --constraint=vram24g&cc8.9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --output=logs/zac_autocts_etth1_%j.out
#SBATCH --error=logs/zac_autocts_etth1_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# ZeroAutoCL — AutoCTS++-aligned single-stage pipeline on ETTh1.
#
# P0 fix (2026-04-29): the previous two-stage noisy/clean curriculum was
# removed because it failed to learn at ZeroAutoCL's data scale (BCE valid
# loss stayed at ~ln 2, accuracy ≈ 0.6 — see CLAUDE_DEBUG.md analysis).
# We now run a SINGLE full-budget seed-gen pass and rely on
# ``forecasting_task_variants`` slicing (3 windows × 3 var-subsets) to
# multiply effective training pairs without extra sampling.
#
# Update (2026-04-30): cross-source L-share pool added (AutoCTS++
# generate_seeds.py L65-103 mechanism).  ``seed_generation.n_shared`` in
# default.yaml controls how many of the per-source candidates are drawn
# from a pool that is identical across every source (default 10 of 20).
# Wallclock unchanged — same number of CL evaluations per (source × sub-
# task), but each shared config now appears in (sources × sub-tasks)
# (task, perf) cells, giving the comparator's task-feature head dense
# task-conditional supervision.
#
# Stages (write into ``${OUT_DIR}``):
#
#   Stage A    : encoder grid (currently SKIPPED — we use the GGS-default
#                encoder for the rest of the pipeline)
#                → encoder_grid.json (synthetic single-row)
#   Stage B.1  : seed generation, full training budget
#                → seeds/seeds.json + per-source ``_seeds_<ds>.json`` checkpoints
#   Stage B.2  : T-CLSC training (single stage, per-task z-scored gap-split)
#                → comparator.pt
#   Stage B.3  : zero-shot ranking on ETTh1
#                → finalists.json
#   Phase 4    : Cartesian retrain (K_enc × top_k_strategies) on ETTh1
#                → phase4_results.json + phase4_best.json
#
# Per-source budget: ``n_per_dataset=20`` candidates × 9 sub-tasks (3 time
# windows × 3 variable subsets, modulo min_var_count gating for narrow
# sources) per source.  Across 6 sources → up to 6 × 20 × 9 = 1080 records.
#
# Wall-clock estimate (RTX 4090, 24 GB):
#   Stage A     :  ~0       (skipped — synthetic encoder_grid.json)
#   Stage B.1   :  ~30–35 h (≈9× the previous single-pool clean run @ 12 h)
#   Stage B.2+3 :  ≤ 1 h
#   Phase 4     :  ~11 h
#                  -------
#                  ≈ 42–46 h
#
# If wall-clock is tight, drop ``forecasting_task_variants.n_variable_subsets``
# back to 1 (cuts seed gen to ~12 h at the cost of ~3× fewer training pairs
# from wide sources).
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

SEED=42

ENCODER_GRID="${OUT_DIR}/encoder_grid.json"
SEEDS_DIR="${OUT_DIR}/seeds"
SEEDS="${SEEDS_DIR}/seeds.json"
COMPARATOR="${OUT_DIR}/comparator.pt"
FINALISTS="${OUT_DIR}/finalists.json"
PHASE4_BEST="${OUT_DIR}/phase4_best.json"

mkdir -p logs "${OUT_DIR}" "${SEEDS_DIR}"

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
# Stage B.1 — seed generation (full training budget, sub-task slicing on)
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Stage B.1: seed generation ═══════"
if [[ -s "${SEEDS}" ]]; then
    echo "[stage-B.1] ${SEEDS} exists — skipping (delete to re-run)"
else
    srun --unbuffered python -u scripts/run_generate_seeds.py \
        --data_dir       "${DATA_DIR}" \
        --save_dir       "${SEEDS_DIR}" \
        --config         "${CONFIG}" \
        --datasets       "${SOURCES[@]}" \
        --n_per_dataset  "${N_PER_DATASET}" \
        --encoder_grid   "${ENCODER_GRID}" \
        --top_k_enc      "${TOP_K_ENC}" \
        --seed           "${SEED}" \
        > >(tee -a "${OUT_DIR}/run.log") 2> >(tee -a "${OUT_DIR}/run.log" >&2)
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage B.2 — T-CLSC comparator training (single stage, z-scored gap split)
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Stage B.2: T-CLSC training ═══════"
if [[ -s "${COMPARATOR}" ]]; then
    echo "[stage-B.2] ${COMPARATOR} exists — skipping (delete to re-run)"
else
    srun --unbuffered python -u scripts/run_pretrain_comparator.py \
        --seeds_path  "${SEEDS}" \
        --data_dir    "${DATA_DIR}" \
        --save_path   "${COMPARATOR}" \
        --config      "${CONFIG}" \
        --seed        "${SEED}" \
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
echo "Seed budget    : ${N_PER_DATASET}/source × ${#SOURCES[@]} sources × n_sub_tasks (set in YAML)"
echo "Outputs:"
echo "  Stage A      : ${ENCODER_GRID}"
echo "  Stage B.1    : ${SEEDS}"
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
