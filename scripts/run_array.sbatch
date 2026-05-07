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
#SBATCH --time=48:00:00
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
# Workflow (fully automatic — one command, self-contained)
# --------------------------------------------------------
#
#   sbatch scripts/run_array.sbatch
#
# That's the entire submission.  Task 0 of the array GENERATES an inline
# downstream sbatch at ``${OUT_DIR}/_chained_post_<arr_jid>.sbatch`` and
# submits it with ``--dependency=afterok:${SLURM_ARRAY_JOB_ID}``, so once
# every array task (B.1 seed-gen for one source) finishes OK, the
# downstream job is automatically launched and runs:
#
#     final seeds merge → Stage T (precompute task features for sources
#     + target) → B.2 (T-CLSC training) → B.3 (zero-shot ranking) →
#     Phase 4 (Cartesian retrain on the target)
#
# — no separate sbatch file, no manual ``precompute_task_features.py``
# invocation, nothing.  Each stage is resume-safe — re-running this
# script after a partial failure picks up where it left off via the
# per-source ``_seeds_<SRC>.json`` checkpoints, the per-dataset
# ``{cache_dir}/<ds>_task_feature.npy`` files, and per-stage output-file
# gates.
#
# Opt out (B.1 only, e.g. for debugging the array):
#   ZAC_NO_CHAIN=1 sbatch scripts/run_array.sbatch
# Re-enable by re-running ``sbatch scripts/run_array.sbatch`` with the env
# var unset; the array tasks resume off their per-source checkpoints and
# task 0 generates+submits a fresh downstream sbatch.
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

# ── Common downstream paths + YAML scalar reads ────────────────────────────
# Defined here (every task) so both the auto-chain block (task 0 only) and
# the per-task Stage B.1 below can share path/scalar definitions.  Doing
# the YAML reads up-front also lets task 0 bake the values into the
# generated downstream sbatch before any seed-gen work starts.
TARGET="ETTh1"
COMPARATOR="${OUT_DIR}/comparator.pt"
FINALISTS="${OUT_DIR}/finalists.json"
PHASE4_BEST="${OUT_DIR}/phase4_best.json"
SEEDS="${SEEDS_DIR}/seeds.json"

N_PER_DATASET=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}', encoding='utf-8')); print(c['two_stage_search']['cl_strategy_search']['n_per_dataset'])")
TOP_K_STR=$(python -c    "import yaml; c=yaml.safe_load(open('${CONFIG}', encoding='utf-8')); print(c['two_stage_search']['cl_strategy_search']['top_k_strategies'])")
TOP_K_ENC=1
echo "YAML scalars     : n_per_dataset=${N_PER_DATASET}  top_k_enc=${TOP_K_ENC} (forced, GGS only)  top_k_strategies=${TOP_K_STR}"

# ── Auto-chain: queue the downstream pipeline upfront ─────────────────────
# Task 0 (only) GENERATES an inline downstream sbatch in ${OUT_DIR} that
# runs B.2 → B.3 → Phase 4, then submits it with
# ``--dependency=afterok:${SLURM_ARRAY_JOB_ID}`` so the entire chain
# (B.1 array → B.2 → B.3 → Phase 4) launches from a single
# ``sbatch scripts/run_array.sbatch`` command.  No external sbatch file
# required — the downstream sbatch is written into ${OUT_DIR} for full
# transparency / debuggability.
#
# We do this BEFORE the heavy seed-gen work so the dependent job is
# visible in ``squeue`` immediately; ``afterok`` semantics keep it PD
# until every array task completes successfully.
#
# Requeue safety: a lock file in OUT_DIR keeps re-runs of task 0 (e.g.
# spot pre-emption) from queuing duplicate downstream jobs.
#
# Opt out for B.1-only (debugging):  ZAC_NO_CHAIN=1 sbatch scripts/run_array.sbatch
LOCK="${OUT_DIR}/.auto_chain_${SLURM_ARRAY_JOB_ID:-manual}"
DEP_SBATCH="${OUT_DIR}/_chained_post_${SLURM_ARRAY_JOB_ID:-manual}.sbatch"

if [[ "${ARRAY_ID}" == "0" && "${ZAC_NO_CHAIN:-}" != "1" ]]; then
    if [[ -f "${LOCK}" ]]; then
        echo "[auto-chain] downstream already queued (lock: ${LOCK})"
        echo "[auto-chain] previous JID: $(cat "${LOCK}" 2>/dev/null || echo "?")"
    else
        # Generate the downstream sbatch.  Heredoc <<DEPEOF (unquoted)
        # expands ${VAR} at SUBMIT time so paths/scalars get baked in;
        # \${VAR} is preserved literally and is resolved when the
        # downstream job actually runs (e.g. SLURM_* runtime vars).
        cat > "${DEP_SBATCH}" <<DEPEOF
#!/usr/bin/env bash
#SBATCH --job-name=zac_post_${TARGET}
#SBATCH --partition=gpu_spot
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --gpu-bind=closest
#SBATCH --constraint=vram24g&cc8.9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/zac_post_${TARGET}_%j.out
#SBATCH --error=logs/zac_post_${TARGET}_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT}"
cd "\${PROJECT_ROOT}"

# Paths / scalars baked at submit-time from the parent run_array.sbatch task.
OUT_DIR="${OUT_DIR}"
DATA_DIR="${DATA_DIR}"
CONFIG="${CONFIG}"
TARGET="${TARGET}"
SEED=${SEED}
ENCODER_GRID="${ENCODER_GRID}"
SEEDS_DIR="${SEEDS_DIR}"
SEEDS="${SEEDS}"
COMPARATOR="${COMPARATOR}"
FINALISTS="${FINALISTS}"
PHASE4_BEST="${PHASE4_BEST}"
TOP_K_ENC=${TOP_K_ENC}
TOP_K_STR=${TOP_K_STR}

module load anaconda
conda activate zeroautocl

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="\${SEED}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-8}"

echo "================================================================="
echo "Downstream job ID : \${SLURM_JOB_ID}"
echo "Node              : \$(hostname)"
echo "Partition         : \${SLURM_JOB_PARTITION:-unknown}"
echo "GPUs visible      : \${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Project root      : \${PROJECT_ROOT}"
echo "Output dir        : \${OUT_DIR}"
echo "Target            : \${TARGET}"
echo "Started           : \$(date -Is)"
echo "================================================================="

nvidia-smi

# Final safety merge — every array task already runs the merger at its end,
# but this re-merge is idempotent and cheap, so we do it once more here so
# that \${SEEDS} reflects every per-source checkpoint regardless of which
# array task was the last writer.
echo
echo "[chained] final seeds merge"
python -u scripts/merge_seed_checkpoints.py \\
    --ckpt_dir "\${SEEDS_DIR}" \\
    --out      "\${SEEDS}" \\
    > >(tee -a "\${OUT_DIR}/run_post.log") \\
    2> >(tee -a "\${OUT_DIR}/run_post.log" >&2)

# ── Stage T — precompute task features for sources + target ────────────────
# AutoCTS++-aligned task features (full TS2Vec encode + sample 100 windows)
# must exist on disk before B.2 — :class:\`TaskFeatureExtractor\` is a pure
# loader, not a trainer.  ``precompute_task_features.py`` is idempotent:
# datasets already cached at \${TASK_FEAT_DIR} are skipped automatically,
# so this step costs ~0 on resume runs.
#
# slide_batch=64 is a safe default for the widest source (PEMS07, 883 ch);
# narrower datasets fit easily and aren't slowed materially.
echo
echo "═══════ Stage T: precompute task features ═══════"
TASK_FEAT_DIR="outputs/task_features"
mkdir -p "\${TASK_FEAT_DIR}"
srun --unbuffered python -u scripts/precompute_task_features.py \\
    --datasets    PEMS03 PEMS04 PEMS07 PEMS08 ETTh2 ExchangeRate "\${TARGET}" \\
    --data_dir    "\${DATA_DIR}" \\
    --cache_dir   "\${TASK_FEAT_DIR}" \\
    --epochs      20 \\
    --slide_batch 64 \\
    --seed        "\${SEED}" \\
    > >(tee -a "\${OUT_DIR}/run_post.log") \\
    2> >(tee -a "\${OUT_DIR}/run_post.log" >&2)

# ── Stage B.2 — T-CLSC training ────────────────────────────────────────────
echo
echo "═══════ Stage B.2: T-CLSC training ═══════"
if [[ -s "\${COMPARATOR}" ]]; then
    echo "[stage-B.2] \${COMPARATOR} exists — skipping (delete to re-run)"
else
    srun --unbuffered python -u scripts/run_pretrain_comparator.py \\
        --seeds_path  "\${SEEDS}" \\
        --data_dir    "\${DATA_DIR}" \\
        --save_path   "\${COMPARATOR}" \\
        --config      "\${CONFIG}" \\
        --seed        "\${SEED}" \\
        > >(tee -a "\${OUT_DIR}/run_post.log") \\
        2> >(tee -a "\${OUT_DIR}/run_post.log" >&2)
fi

# ── Stage B.3 — zero-shot ranking on target ────────────────────────────────
echo
echo "═══════ Stage B.3: zero-shot ranking on \${TARGET} ═══════"
if [[ -s "\${FINALISTS}" ]]; then
    echo "[stage-B.3] \${FINALISTS} exists — skipping (delete to re-run)"
else
    srun --unbuffered python -u scripts/run_zero_shot_search.py \\
        --comparator_path  "\${COMPARATOR}" \\
        --data_dir         "\${DATA_DIR}" \\
        --target_dataset   "\${TARGET}" \\
        --config           "\${CONFIG}" \\
        --encoder_grid     "\${ENCODER_GRID}" \\
        --top_k_enc        "\${TOP_K_ENC}" \\
        --top_k_strategies "\${TOP_K_STR}" \\
        --output           "\${FINALISTS}" \\
        --seed             "\${SEED}" \\
        > >(tee -a "\${OUT_DIR}/run_post.log") \\
        2> >(tee -a "\${OUT_DIR}/run_post.log" >&2)
fi

# ── Phase 4 — Cartesian retrain on target ──────────────────────────────────
echo
echo "═══════ Phase 4: Cartesian retrain on \${TARGET} ═══════"
if [[ -s "\${PHASE4_BEST}" ]]; then
    echo "[phase4] \${PHASE4_BEST} exists — skipping (delete to re-run)"
else
    srun --unbuffered python -u scripts/run_phase4_finalists.py \\
        --finalists "\${FINALISTS}" \\
        --target    "\${TARGET}" \\
        --data_dir  "\${DATA_DIR}" \\
        --config    "\${CONFIG}" \\
        --save_dir  "\${OUT_DIR}" \\
        --seed      "\${SEED}" \\
        > >(tee -a "\${OUT_DIR}/run_post.log") \\
        2> >(tee -a "\${OUT_DIR}/run_post.log" >&2)
fi

echo
echo "================================================================="
echo "Downstream finished : \$(date -Is)"
echo "Outputs:"
echo "  Stage B.1     : \${SEEDS}"
echo "  Stage T       : \${TASK_FEAT_DIR}/<dataset>_task_feature.npy"
echo "  Stage B.2     : \${COMPARATOR}"
echo "  Stage B.3     : \${FINALISTS}"
echo "  Phase 4       : \${OUT_DIR}/phase4_results.json"
echo "  Best          : \${PHASE4_BEST}"
echo "  Full log      : \${OUT_DIR}/run_post.log"
echo "================================================================="

if [[ -s "\${PHASE4_BEST}" ]]; then
    echo
    echo ">>> Best finalist:"
    python -c "
import json
with open('\${PHASE4_BEST}') as f:
    b = json.load(f)
print(f\"  Tag      : {b.get('tag', 'n/a')}\")
for k in ('mean_mse', 'mean_mae', 'mean_mse_raw', 'mean_mae_raw'):
    if k in b:
        print(f\"  {k:9s}: {b[k]:.4f}\")
print(f\"  Encoder  : {b.get('encoder_config', 'n/a')}\")
"
fi
DEPEOF

        # Submit downstream with afterok dep on the entire array.
        # Fail-safe: a flaky compute-node sbatch invocation must NOT mark
        # the B.1 task FAILED.  Disable pipefail locally and capture rc.
        set +e
        DEP_OUT=$(sbatch --parsable \
            --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
            "${DEP_SBATCH}" 2>&1)
        DEP_RC=$?
        set -e
        if [[ ${DEP_RC} -ne 0 ]]; then
            echo "[auto-chain] WARNING: sbatch failed (rc=${DEP_RC}): ${DEP_OUT}"
            echo "[auto-chain] proceeding with seed-gen — submit downstream manually:"
            echo "             sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} ${DEP_SBATCH}"
        else
            DEP_JID="${DEP_OUT}"
            echo "${DEP_JID}" > "${LOCK}"
            echo "[auto-chain] generated downstream sbatch: ${DEP_SBATCH}"
            echo "[auto-chain] queued downstream job ${DEP_JID}"
            echo "             will run when all tasks of array ${SLURM_ARRAY_JOB_ID} finish OK"
            echo "             ('scancel ${SLURM_ARRAY_JOB_ID} ${DEP_JID}' to abort the whole chain)"
        fi
    fi
elif [[ "${ZAC_NO_CHAIN:-}" == "1" ]]; then
    [[ "${ARRAY_ID}" == "0" ]] && echo "[auto-chain] disabled by ZAC_NO_CHAIN=1 — submit downstream manually"
fi

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
# Per-task post-merge:  every array task ends by re-merging whatever
# ``_seeds_*.json`` checkpoints currently exist into ``seeds.json``.
#
# Why every task and not just task 0:
#   ``run_generate_seeds.py`` internally calls ``generate_seeds(...)`` which,
#   at the end of the function, also writes ``seeds.json`` — but only with
#   the records this task processed (a single source, in array mode).  All
#   six tasks race to write the same path, last writer wins, and the final
#   ``seeds.json`` ends up holding exactly one source's records.  Adding a
#   merge step at the end of every task fixes this without coordination:
#
#     - The merger is idempotent (atomic temp-rename, no partial state).
#     - Each task's merger reads every ``_seeds_*.json`` present at that
#       moment, including its own (just written above).
#     - The LAST array task to finish runs its merger after all six
#       checkpoints exist, so the final ``seeds.json`` is always correct.
#     - Earlier mergers may write incomplete content, but each subsequent
#       writer (seed-gen step 2 OR another task's merger) overwrites it.
#
# The downstream sbatch (auto-generated by task 0 — see auto-chain block
# above) only fires under ``afterok:${SLURM_ARRAY_JOB_ID}`` so it is
# guaranteed to read the post-last-merger state.  It also re-runs the
# merger itself once for full safety before B.2.
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "[stage-B.1] [${SRC}] merging per-source checkpoints into ${SEEDS_DIR}/seeds.json"
python -u scripts/merge_seed_checkpoints.py \
    --ckpt_dir "${SEEDS_DIR}" \
    --out      "${SEEDS_DIR}/seeds.json" \
    > >(tee -a "${OUT_DIR}/run_array_${ARRAY_ID}_${SRC}.log") \
    2> >(tee -a "${OUT_DIR}/run_array_${ARRAY_ID}_${SRC}.log" >&2)

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
if [[ -f "${LOCK}" ]]; then
    echo "Downstream       : auto-chained, JID $(cat "${LOCK}" 2>/dev/null || echo "?")"
    echo "                   sbatch script   : ${DEP_SBATCH}"
    echo "                   (waits on afterok:${SLURM_ARRAY_JOB_ID:-?}; runs B.2 → B.3 → Phase 4)"
else
    echo "Downstream       : NOT auto-chained (ZAC_NO_CHAIN=1 or task != 0)"
    echo "                   re-run with chain enabled:  sbatch scripts/run_array.sbatch"
fi
echo "================================================================="
