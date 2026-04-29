#!/usr/bin/env bash
#SBATCH --job-name=zac_array_seedgen
#SBATCH --partition=gpu_spot
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --gpu-bind=closest
#SBATCH --constraint=vram24g&cc8.9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-5%6
#SBATCH --output=logs/zac_array_seedgen_%A_%a.out
#SBATCH --error=logs/zac_array_seedgen_%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# ZeroAutoCL — Stage B.1 (seed generation) parallelised across SLURM job array.
#
# One array task per source dataset: each task does a full seed-gen run for
# its own source and writes ``${SEEDS_DIR}/_seeds_<SRC>.json``.  The 6 tasks
# run concurrently (``--array=0-5%6``); per-source CL pretraining is the
# real bottleneck (~30-35 h sequential), so 6-way fan-out cuts wallclock to
# the slowest source's per-source time (~6-8 h, dominated by PEMS07's
# 883-channel I/O).
#
# L-share consistency is preserved across the array thanks to two pieces:
#
#   1. The shared 10 candidates are sampled by ``set_seed(seed)`` only —
#      identical across all 6 tasks regardless of array task ID.
#   2. ``--source_global_idx_offset $SLURM_ARRAY_TASK_ID`` is threaded into
#      ``generate_seeds(...)`` so each task's per-source random tail seed
#      and per-candidate training seed line up exactly with the sequential
#      single-job ordering.  Verified byte-equal against sequential mode.
#
# Workflow
# --------
#
#   # 1) Submit the array (6 parallel jobs, one per source)
#   JID=$(sbatch --parsable scripts/run_array.sbatch)
#
#   # 2) Submit the dependent finish job (B.2 → Phase 4) once all
#   #    array tasks finish OK.  ``run_full_etth1_autocts.sbatch``'s B.1
#   #    stage will see all ``_seeds_<SRC>.json`` checkpoints and skip
#   #    retraining; it just merges them into ``seeds.json`` and proceeds.
#   sbatch --dependency=afterok:${JID} scripts/run_full_etth1_autocts.sbatch
#
# Tightening %N
# -------------
# ``--array=0-5%6`` allows up to 6 tasks concurrent.  If the cluster is
# tight or spot pre-emption is aggressive, drop to ``%3`` or ``%2`` — total
# wallclock will rise but throughput stays robust.
#
# Submit  :  sbatch scripts/run_array.sbatch
# Watch   :  tail -f logs/zac_array_seedgen_<jobid>_<task>.out
# Cancel  :  scancel <jobid>           # cancels whole array
# Cancel one task : scancel <jobid>_<task>
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/ZeroAutoCL}"
cd "$PROJECT_ROOT"

OUT_DIR="outputs/full_etth1_autocts"
DATA_DIR="data/datasets"
CONFIG="configs/default.yaml"

SEED=42

ENCODER_GRID="${OUT_DIR}/encoder_grid.json"
SEEDS_DIR="${OUT_DIR}/seeds"

mkdir -p logs "${OUT_DIR}" "${SEEDS_DIR}"

# Source list — index = SLURM_ARRAY_TASK_ID.  PEMS07 is task 0 because it
# is the heaviest source (883 sensor channels) and the most likely to OOM
# or saturate wall-clock; failing fast here lets the user catch resource
# issues without waiting on the lighter tasks.
SOURCES=(PEMS07 ETTh2 PEMS03 PEMS04 PEMS08 ExchangeRate)

ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( ARRAY_ID < 0 || ARRAY_ID >= ${#SOURCES[@]} )); then
    echo "[error] SLURM_ARRAY_TASK_ID=${ARRAY_ID} out of range [0, ${#SOURCES[@]})"
    exit 1
fi
SRC="${SOURCES[${ARRAY_ID}]}"

echo "================================================================="
echo "Array job ID     : ${SLURM_ARRAY_JOB_ID:-unknown}"
echo "Array task ID    : ${ARRAY_ID}"
echo "Source           : ${SRC}"
echo "Node             : $(hostname)"
echo "Partition        : ${SLURM_JOB_PARTITION:-unknown}"
echo "GPUs visible     : ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Project root     : ${PROJECT_ROOT}"
echo "Output dir       : ${OUT_DIR}"
echo "Seeds dir        : ${SEEDS_DIR}"
echo "Started          : $(date -Is)"
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

# ── Synthetic encoder_grid.json (Stage A is SKIP — we use GGS encoder) ─────
# Each array task ensures this file exists.  If it already does (e.g. another
# task wrote it), skip — atomic write → no race risk.
if [[ ! -s "${ENCODER_GRID}" ]]; then
    python - <<EOF
import json, yaml, os
out = "${ENCODER_GRID}"
if os.path.exists(out) and os.path.getsize(out) > 0:
    print(f"[stage-A] {out} already populated — skipping write")
else:
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
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp, out)
    print(f"[stage-A] wrote synthetic encoder_grid.json with GGS-only encoder: {records[0]['encoder_config']}")
EOF
fi

# Read scalars from YAML once.
N_PER_DATASET=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}', encoding='utf-8')); print(c['two_stage_search']['cl_strategy_search']['n_per_dataset'])")
TOP_K_ENC=1
echo "YAML scalars     : n_per_dataset=${N_PER_DATASET}  top_k_enc=${TOP_K_ENC} (forced, GGS only)"

# ─────────────────────────────────────────────────────────────────────────────
# Stage B.1 — seed generation for ONE source (this array task's slice)
# ─────────────────────────────────────────────────────────────────────────────
SRC_CKPT="${SEEDS_DIR}/_seeds_${SRC}.json"

echo
echo "═══════ Stage B.1 [${SRC}] (array task ${ARRAY_ID}) ═══════"
if [[ -s "${SRC_CKPT}" ]]; then
    echo "[stage-B.1] ${SRC_CKPT} exists — skipping (delete to re-run)"
else
    srun --unbuffered python -u scripts/run_generate_seeds.py \
        --data_dir                  "${DATA_DIR}" \
        --save_dir                  "${SEEDS_DIR}" \
        --config                    "${CONFIG}" \
        --datasets                  "${SRC}" \
        --n_per_dataset             "${N_PER_DATASET}" \
        --encoder_grid              "${ENCODER_GRID}" \
        --top_k_enc                 "${TOP_K_ENC}" \
        --source_global_idx_offset  "${ARRAY_ID}" \
        --seed                      "${SEED}" \
        > >(tee -a "${OUT_DIR}/run_array_${ARRAY_ID}_${SRC}.log") \
        2> >(tee -a "${OUT_DIR}/run_array_${ARRAY_ID}_${SRC}.log" >&2)
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "================================================================="
echo "Finished         : $(date -Is)"
echo "Source           : ${SRC}"
echo "Checkpoint       : ${SRC_CKPT}"
if [[ -s "${SRC_CKPT}" ]]; then
    NREC=$(python -c "import json; print(len(json.load(open('${SRC_CKPT}'))['records']))" 2>/dev/null || echo "?")
    echo "Records          : ${NREC}"
fi
echo "Log              : ${OUT_DIR}/run_array_${ARRAY_ID}_${SRC}.log"
echo
echo "Next step:"
echo "  After all 6 array tasks finish OK, submit the downstream sbatch:"
echo "    sbatch --dependency=afterok:\${SLURM_ARRAY_JOB_ID} \\"
echo "        scripts/run_full_etth1_autocts.sbatch"
echo "  ('run_full_etth1_autocts.sbatch' will detect all _seeds_<SRC>.json"
echo "   checkpoints, skip B.1 retraining, merge into seeds.json, then run"
echo "   B.2 → B.3 → Phase 4.)"
echo "================================================================="
