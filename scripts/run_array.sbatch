#!/usr/bin/env bash
#SBATCH --job-name=zac_array_seedgen
#SBATCH --partition=gpu_spot
#SBATCH --gpus=1
#SBATCH --gpu-bind=closest
# GPU pool (rev 2026-05-20, 4090 + 3090):
#
# Eligible GPU on ``gpu_spot``:
#   • ``nvidia_geforce_rtx_4090`` 24GB  CC 8.9   g1n[1-6]            6 × 1 GPU = 6
#   • ``nvidia_geforce_rtx_3090`` 24GB  CC 8.6   g2n1 + g4n[1-2]     1×2 + 2×1 = 4
#   Pool size: 10 GPUs.
#
# Excluded (still — user request 2026-05-16):
#   • A100  80GB CC 8.0   (vram80g != vram24g; also no cc8.6 feature)
#   • L40s  48GB CC 8.9   (vram48g != vram24g)
#
# Constraint mechanism:
#   ``--constraint=vram24g&cc8.6`` requires BOTH features.  In this
#   cluster, ``vram*`` features are exact-tier (vram24g != vram48g) but
#   ``cc*`` features are cumulative on the same node — 4090 nodes
#   advertise cc8.0, cc8.6 AND cc8.9 simultaneously (verified via
#   ``sinfo -N -p gpu_spot -o "%n %f"`` on 2026-05-20).  So
#   ``vram24g&cc8.6`` matches 4090 (CC 8.9 ⊃ cc8.6 feature) and 3090
#   (CC 8.6 native), while excluding L40s (no vram24g) and A100 (no
#   cc8.6 AND no vram24g).
#
# Concurrency implication:
#   10 GPUs in the pool.  With ``--array=0-8%9`` all 9 tasks can run in
#   parallel when the pool is uncontended (wallclock ≈ slowest single
#   source).  3090 fallback prevents the 2-of-6 4090-contention bottleneck
#   observed on 2026-05-20 (4 of 6 4090 nodes were chronically allocated
#   to other tenants, throttling us to 2 concurrent tasks).
#
# Rev 2026-05-20 change (user request): relaxed cc8.9 → cc8.6 to include
# 3090.  L40s / A100 still excluded — those exclusions are vram-driven,
# not cc-driven, so this loosening doesn't accidentally re-enable them.
#
# The ``nvidia-smi`` printout at job start (further down) confirms the
# assigned GPU model.
#SBATCH --constraint=vram24g&cc8.6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --array=0-8%9
#SBATCH --output=logs/zac_array_seedgen_%A_%a.out
#SBATCH --error=logs/zac_array_seedgen_%A_%a.err

# ─────────────────────────────────────────────────────────────────────────────
# ZeroAutoCL — Stage B.1 (seed generation) parallelised across SLURM job array.
# Rev 2026-05-20: 9-source CL-aligned pool + 3 targets (ETTh1, Electricity,
# Weather).  ONCE STARTED, NO MANUAL OPS — the auto-chain in task 0 queues
# the entire downstream pipeline (Stage T → B.2 → 3× (B.3 + Phase 4)).
#
# One array task per source dataset: each task does a full seed-gen run for
# its own source and writes ``${SEEDS_DIR}/_seeds_<SRC>.json``.  9 tasks
# total (``--array=0-8%9``), ALL concurrent — the 10-GPU pool (4090+3090) on
# ``gpu_spot`` comfortably absorbs 9 simultaneous jobs (1 per source).
#
# Rev 2026-05-20 — AutoCTS++ noisy-only alignment (T12+T13) + time-window
# randomization (rev 2026-05-19) + adaptive iter:
#   - Clean stage DISABLED (yaml ``n_per_dataset=0``); random-arch clean
#     labels were AutoCTS++ dead code (``mode='clean_seeds'`` never
#     invoked).  Phase 4 retrain is the only clean-equivalent training
#     (mirrors AutoCTS++ ``mode='train'``).
#   - Noisy stage ONLY: ``n_noisy_per_dataset=80`` per sub-task, with
#     per-source adaptive iter budgets (ETTh2: 388, others: 480 — see
#     ``adaptive_iter.log_clip`` block in default.yaml) and best-of-N=5
#     over checkpoints (label = max over the 5 perfs).
#   - L-share: ``n_shared=40`` (50/50 of n_noisy=80, AutoCTS++ Algorithm 1
#     ``|S0|=|Si|=L``); shared head identical across every source.
#   - Time-window randomization: per-source random (start, length) windows
#     with IoU≤0.7 rejection → 111 sub-tasks total (was 57 equal-length).
#     Per-source params in ``forecasting_task_variants.time_window_params``.
#
# Per-(sub_idx, stage) checkpointing inside ``seed_generator.py`` limits
# spot-pre-emption loss to one (sub-task × stage) chunk (~30 min worst case).
# Downstream after array completes: Stage T 1h + B.2 30min + 3 targets ×
# ~1.5h each ≈ 6h.  Grand total wall ≈ 3 days single-card → ~1 day with
# 9-wide parallel array (driven by the slowest source — traffic or Solar).
#
# L-share consistency is preserved across the array thanks to two pieces:
#
#   1. The shared 40 candidates (``n_shared=40`` in yaml) are sampled by
#      ``set_seed(seed)`` only — identical across all 9 tasks regardless
#      of array task ID.
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
#     final seeds merge → Stage T (precompute task features for the 9
#     sources + 3 targets) → B.2 (T-CLSC training, one shared comparator)
#     → for each of 3 targets {ETTh1, Electricity, Weather}: B.3
#     (zero-shot ranking) → Phase 4 (Cartesian retrain on that target).
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
# ``--array=0-8%9`` allows up to 9 tasks concurrent (= total tasks → full
# parallel).  If the cluster is tight or spot pre-emption is aggressive,
# drop to ``%6`` / ``%3`` / ``%2`` — total wallclock will rise but
# throughput stays robust.  At %9, wallclock ≈ slowest single source.
#
# Submit  :  sbatch scripts/run_array.sbatch
# Watch   :  tail -f logs/zac_array_seedgen_<jobid>_<task>.out
# Cancel  :  scancel <jobid>           # cancels whole array
# Cancel one task : scancel <jobid>_<task>
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/ZeroAutoCL}"
cd "$PROJECT_ROOT"

OUT_DIR="outputs/zac_2026_05_20"
DATA_DIR="data/datasets"
CONFIG="configs/default.yaml"

SEED=42

ENCODER_GRID="${OUT_DIR}/encoder_grid.json"
SEEDS_DIR="${OUT_DIR}/seeds"

mkdir -p logs "${OUT_DIR}" "${SEEDS_DIR}"

# Source list — index = SLURM_ARRAY_TASK_ID.  traffic is task 0 because it
# has the most channels (862) and is the most likely to OOM or saturate
# wallclock; failing fast here lets the user catch resource issues without
# waiting on the lighter tasks.
#
# 9-source CL-aligned pool (rev 2026-05-20, time-window randomization +
# AQWanliu boost) — every target has ≥2 same-domain neighbours.  Per-source
# random-window params (n_windows, min_len, max_len) live in
# ``forecasting_task_variants.time_window_params`` (default.yaml).  Windows
# are sampled uniformly in [min_len, max_len] with pairwise IoU ≤ 0.7, and
# every min_len ≥ encoder RF (2047) to keep CL pretraining stable:
#
#   ETT family   (univariate OT column, auto-skips var-subsetting):
#     - ETTh2   6 × 1 =  6 sub-tasks   (window 5800-12000, hourly)
#     - ETTm1  15 × 1 = 15 sub-tasks   (window 5800-25000, 15-min)
#     - ETTm2  15 × 1 = 15 sub-tasks   (window 5800-25000, 15-min)
#   Energy:
#     - Solar    137 ch,  6 × 3 = 18 sub-tasks   (window 5800-20000, 10-min)
#   Transport (no graph):
#     - traffic  862 ch,  6 × 3 = 18 sub-tasks   (window 5800-12000, hourly)
#   Atmospheric (var-subsetting disabled to preserve pollutant×weather):
#     - AQShunyi    11 ch,  6 × 1 =  6 sub-tasks   (window 5800-25000, hourly)
#     - AQWanliu    11 ch, 12 × 1 = 12 sub-tasks   (boost: cleanest noisy ρ)
#     - AQGuanyuan  11 ch,  6 × 1 =  6 sub-tasks
#   Finance:
#     - ExchangeRate  8 ch,  5 × 3 = 15 sub-tasks  (window 4096-7588, daily)
#
# Total 111 sub-tasks.  Domain breakdown: ETT 32% / Energy 16% / Transport
# 16% / Atmospheric 22% (AQWanliu boosted to feed Weather target) /
# Finance 14% — broadly mirrors AutoCTS++ §4.1.1's 36% ETT ratio with
# intentional +6pp on Atmospheric (Weather has no in-domain neighbour
# outside the 3 Beijing PRSA stations).
#
# Order: heaviest first (traffic, Solar) → ETT-15min → finance → AQ
# stations → ETTh2 (cheapest).  Failing fast on resource issues at task 0.
SOURCES=(traffic Solar ETTm1 ETTm2 ExchangeRate AQShunyi AQWanliu AQGuanyuan ETTh2)

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
# Three targets — Stage B.3 + Phase 4 run sequentially per target inside
# the downstream sbatch (one shared comparator from Stage B.2 serves all
# three).  Per-target outputs live in ``${OUT_DIR}/finalists_<target>.json``
# and ``${OUT_DIR}/phase4_<target>/phase4_best.json``.
TARGETS=(ETTh1 Electricity Weather)
COMPARATOR="${OUT_DIR}/comparator.pt"
SEEDS="${SEEDS_DIR}/seeds.json"

# Rev 2026-05-12: read clean ``n_per_dataset`` from ``seed_generation``
# (the canonical key actually consumed by ``run_generate_seeds.py``).  The
# previous read from ``two_stage_search.cl_strategy_search.n_per_dataset``
# silently overrode the seed-generation value via CLI — bug.  Also read the
# noisy-stage knobs so they're transparent in the log and explicitly piped
# to the CLI rather than silently default-loaded.
N_PER_DATASET=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}', encoding='utf-8')); print(c['seed_generation']['n_per_dataset'])")
N_NOISY_PER_DATASET=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}', encoding='utf-8')); print(c['seed_generation'].get('n_noisy_per_dataset', 0) or 0)")
NOISY_ITERS=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}', encoding='utf-8')); print(c['seed_generation'].get('noisy_pretrain_iters', 100) or 100)")
TOP_K_STR=$(python -c    "import yaml; c=yaml.safe_load(open('${CONFIG}', encoding='utf-8')); print(c['two_stage_search']['cl_strategy_search']['top_k_strategies'])")
TOP_K_ENC=1
echo "YAML scalars     : n_per_dataset=${N_PER_DATASET} (clean)  n_noisy_per_dataset=${N_NOISY_PER_DATASET}  noisy_pretrain_iters=${NOISY_ITERS}  top_k_enc=${TOP_K_ENC} (forced, GGS only)  top_k_strategies=${TOP_K_STR}"

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
#SBATCH --job-name=zac_post_multi
#SBATCH --partition=gpu_spot
#SBATCH --gpus=1
#SBATCH --gpu-bind=closest
# Mirror the parent run_array.sbatch GPU restriction: 4090 + 3090
# (24 GB VRAM, CC >= 8.6).  Pool size = 10 (g1n[1-6] + g2n1 + g4n[1-2]).
# See parent banner for the rev 2026-05-20 rationale (user request).
#SBATCH --constraint=vram24g&cc8.6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/zac_post_multi_%j.out
#SBATCH --error=logs/zac_post_multi_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT}"
cd "\${PROJECT_ROOT}"

# Paths / scalars baked at submit-time from the parent run_array.sbatch task.
OUT_DIR="${OUT_DIR}"
DATA_DIR="${DATA_DIR}"
CONFIG="${CONFIG}"
TARGETS=(ETTh1 Electricity Weather)
SEED=${SEED}
ENCODER_GRID="${ENCODER_GRID}"
SEEDS_DIR="${SEEDS_DIR}"
SEEDS="${SEEDS}"
COMPARATOR="${COMPARATOR}"
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
echo "Targets           : \${TARGETS[*]}"
echo "Started           : \$(date -Is)"
echo "================================================================="

nvidia-smi

# Final safety merge — every array task already runs the merger at its end,
# but this re-merge is idempotent and cheap, so we do it once more here so
# that \${SEEDS} reflects every per-source checkpoint regardless of which
# array task was the last writer.
echo
echo "[chained] final seeds merge (with ρ back-fill from logs)"
# --log_dir back-fills any missing noisy↔clean Spearman ρ values from
# run_array_*_<SRC>.log into the per-source checkpoints, then writes
# seeds_meta.json (sibling of seeds.json) so the comparator trainer can
# apply ``comparator.noisy_rho_threshold``.  Idempotent — no-op once
# every checkpoint already carries its ρ values.
python -u scripts/merge_seed_checkpoints.py \\
    --ckpt_dir "\${SEEDS_DIR}" \\
    --out      "\${SEEDS}" \\
    --log_dir  "\${OUT_DIR}" \\
    > >(tee -a "\${OUT_DIR}/run_post.log") \\
    2> >(tee -a "\${OUT_DIR}/run_post.log" >&2)

# ── Stage T — precompute task features for sources + target ────────────────
# AutoCTS++-aligned task features (full TS2Vec encode + sample 100 windows)
# must exist on disk before B.2 — :class:\`TaskFeatureExtractor\` is a pure
# loader, not a trainer.  ``precompute_task_features.py`` is idempotent:
# datasets already cached at \${TASK_FEAT_DIR} are skipped automatically,
# so this step costs ~0 on resume runs.
#
# Iter-based budget (--iters 600) is mandatory for ZeroAutoCL: the
# stride-1 sliding-window TimeSeriesDataset inflates len(dataset) to
# T - 3000 + 1 (~14k samples for traffic at T=17544), so epoch-based
# budgets balloon into 4+ hours per source.  600 iters mirrors TS2Vec's
# n_iters default and bounds Stage T to ~5 min per dataset.
#
# slide_batch=64 is a safe default for the widest source in this pool
# (traffic.csv, 862 ch); narrower datasets fit easily and aren't slowed
# materially.
echo
echo "═══════ Stage T: precompute task features ═══════"
TASK_FEAT_DIR="outputs/task_features"
# Eviction marker for rev 2026-05-20 — time-window randomization (rev
# 2026-05-19) keeps the sub-task slug names (e.g. ETTh2__tw0__vs1)
# identical but changes the underlying window coordinates.
# precompute_task_features.py checks cache hits by FILENAME only, so
# without this guard old .npy files from the equal-window era would be
# silently reused with mismatched data — task features would not match
# the seed records.  Idempotent: presence of the marker file means the
# cache is already on the current rev; subsequent runs skip eviction.
TASK_FEAT_REV_MARKER="\${TASK_FEAT_DIR}/.rev_2026_05_20"
if [[ -d "\${TASK_FEAT_DIR}" && ! -f "\${TASK_FEAT_REV_MARKER}" ]]; then
    echo "[migration] task_features cache predates rev 2026-05-20 — evicting"
    rm -rf "\${TASK_FEAT_DIR}"
fi
mkdir -p "\${TASK_FEAT_DIR}"
touch "\${TASK_FEAT_REV_MARKER}"

# Option B (2026-05-10): sources use --sub_task_mode so each (tw, vs)
# sub-task gets its own task feature .npy (mirrors AutoCTS++'s per-subset
# generate_task_feature.py invocation).  Targets stay in base mode — at
# zero-shot inference time each target is a single task with no sub-task
# decomposition.

echo
echo "─── Stage T.1: sources (sub-task mode, per-(tw, vs) features) ───"
srun --unbuffered python -u scripts/precompute_task_features.py \\
    --datasets    ETTh2 ETTm1 ETTm2 \\
                  Solar traffic ExchangeRate \\
                  AQShunyi AQWanliu AQGuanyuan \\
    --sub_task_mode \\
    --config      "\${CONFIG}" \\
    --data_dir    "\${DATA_DIR}" \\
    --cache_dir   "\${TASK_FEAT_DIR}" \\
    --iters       600 \\
    --slide_batch 64 \\
    --seed        "\${SEED}" \\
    > >(tee -a "\${OUT_DIR}/run_post.log") \\
    2> >(tee -a "\${OUT_DIR}/run_post.log" >&2)

echo
echo "─── Stage T.2: targets (base mode, one feature per dataset) ───"
srun --unbuffered python -u scripts/precompute_task_features.py \\
    --datasets    "\${TARGETS[@]}" \\
    --data_dir    "\${DATA_DIR}" \\
    --cache_dir   "\${TASK_FEAT_DIR}" \\
    --iters       600 \\
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

# ── Per-target zero-shot ranking + Phase 4 retrain (3 targets) ─────────────
# B.3 + Phase 4 run sequentially per target.  The comparator from B.2 is
# shared across all three.  Each target gets its own ``finalists_<t>.json``
# and ``phase4_<t>/phase4_best.json`` so resume is per-target — if one
# target's Phase 4 dies mid-way, re-running the downstream sbatch picks up
# from the next target without redoing completed ones.
for target in "\${TARGETS[@]}"; do
    echo
    echo "═══════════════════════════════════════════════════════════════════"
    echo "Target: \${target}"
    echo "═══════════════════════════════════════════════════════════════════"

    FINALISTS="\${OUT_DIR}/finalists_\${target}.json"
    PHASE4_DIR="\${OUT_DIR}/phase4_\${target}"
    PHASE4_BEST="\${PHASE4_DIR}/phase4_best.json"
    mkdir -p "\${PHASE4_DIR}"

    # Stage B.3 — zero-shot ranking on this target
    echo
    echo "═══════ Stage B.3: zero-shot ranking on \${target} ═══════"
    if [[ -s "\${FINALISTS}" ]]; then
        echo "[stage-B.3] \${FINALISTS} exists — skipping (delete to re-run)"
    else
        srun --unbuffered python -u scripts/run_zero_shot_search.py \\
            --comparator_path  "\${COMPARATOR}" \\
            --data_dir         "\${DATA_DIR}" \\
            --target_dataset   "\${target}" \\
            --config           "\${CONFIG}" \\
            --encoder_grid     "\${ENCODER_GRID}" \\
            --top_k_enc        "\${TOP_K_ENC}" \\
            --top_k_strategies "\${TOP_K_STR}" \\
            --output           "\${FINALISTS}" \\
            --seed             "\${SEED}" \\
            > >(tee -a "\${OUT_DIR}/run_post.log") \\
            2> >(tee -a "\${OUT_DIR}/run_post.log" >&2)
    fi

    # Phase 4 — Cartesian retrain on this target
    echo
    echo "═══════ Phase 4: Cartesian retrain on \${target} ═══════"
    if [[ -s "\${PHASE4_BEST}" ]]; then
        echo "[phase4] \${PHASE4_BEST} exists — skipping (delete to re-run)"
    else
        srun --unbuffered python -u scripts/run_phase4_finalists.py \\
            --finalists "\${FINALISTS}" \\
            --target    "\${target}" \\
            --data_dir  "\${DATA_DIR}" \\
            --config    "\${CONFIG}" \\
            --save_dir  "\${PHASE4_DIR}" \\
            --seed      "\${SEED}" \\
            > >(tee -a "\${OUT_DIR}/run_post.log") \\
            2> >(tee -a "\${OUT_DIR}/run_post.log" >&2)
    fi
done

echo
echo "================================================================="
echo "Downstream finished : \$(date -Is)"
echo "Outputs:"
echo "  Stage B.1     : \${SEEDS}"
echo "  Stage T       : \${TASK_FEAT_DIR}/<dataset>_task_feature.npy"
echo "  Stage B.2     : \${COMPARATOR}"
for t in "\${TARGETS[@]}"; do
    echo "  [\${t}] B.3   : \${OUT_DIR}/finalists_\${t}.json"
    echo "  [\${t}] Ph4   : \${OUT_DIR}/phase4_\${t}/phase4_best.json"
done
echo "  Full log      : \${OUT_DIR}/run_post.log"
echo "================================================================="

echo
echo ">>> Best finalist per target:"
for t in "\${TARGETS[@]}"; do
    best="\${OUT_DIR}/phase4_\${t}/phase4_best.json"
    if [[ -s "\${best}" ]]; then
        echo
        echo "--- \${t} ---"
        python -c "
import json
with open('\${best}') as f:
    b = json.load(f)
print(f\"  Tag      : {b.get('tag', 'n/a')}\")
for k in ('mean_mse', 'mean_mae', 'mean_mse_raw', 'mean_mae_raw'):
    if k in b:
        print(f\"  {k:9s}: {b[k]:.4f}\")
print(f\"  Encoder  : {b.get('encoder_config', 'n/a')}\")
"
    else
        echo "  [\${t}] no phase4_best.json — check log"
    fi
done
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

# Rev 2026-05-12: the per-(sub_idx, stage) checkpoint inside
# ``seed_generator.py`` now handles partial resume itself, so a non-empty
# ``${SRC_CKPT}`` no longer means "done" — it may also mean "partial".  We
# pass through to ``run_generate_seeds.py`` unconditionally; the script's
# own logic skips already-completed (sub_idx, stage) pairs and re-runs the
# rest.  Stale ``n_per_dataset`` / ``n_noisy_per_dataset`` changes auto-
# invalidate the cache so config edits don't require manual deletion.
echo
echo "═══════ Stage B.1 [${SRC}] (array task ${ARRAY_ID}) ═══════"
if [[ -s "${SRC_CKPT}" ]]; then
    echo "[stage-B.1] ${SRC_CKPT} exists — entering partial-resume mode"
    echo "            (delete the file to force a full restart)"
else
    echo "[stage-B.1] no checkpoint, starting fresh"
fi
srun --unbuffered python -u scripts/run_generate_seeds.py \
    --data_dir                  "${DATA_DIR}" \
    --save_dir                  "${SEEDS_DIR}" \
    --config                    "${CONFIG}" \
    --datasets                  "${SRC}" \
    --n_per_dataset             "${N_PER_DATASET}" \
    --n_noisy_per_dataset       "${N_NOISY_PER_DATASET}" \
    --noisy_pretrain_iters      "${NOISY_ITERS}" \
    --encoder_grid              "${ENCODER_GRID}" \
    --top_k_enc                 "${TOP_K_ENC}" \
    --source_global_idx_offset  "${ARRAY_ID}" \
    --seed                      "${SEED}" \
    > >(tee -a "${OUT_DIR}/run_array_${ARRAY_ID}_${SRC}.log") \
    2> >(tee -a "${OUT_DIR}/run_array_${ARRAY_ID}_${SRC}.log" >&2)

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
# --log_dir back-fills noisy↔clean Spearman ρ from logs into per-source
# checkpoints (idempotent) and writes seeds_meta.json next to seeds.json
# so the downstream comparator trainer can apply the noisy-bucket quality
# filter (configs/default.yaml::comparator.noisy_rho_threshold).
python -u scripts/merge_seed_checkpoints.py \
    --ckpt_dir "${SEEDS_DIR}" \
    --out      "${SEEDS_DIR}/seeds.json" \
    --log_dir  "${OUT_DIR}" \
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
    echo "                   (waits on afterok:${SLURM_ARRAY_JOB_ID:-?}; runs Stage T → B.2 → 3×(B.3 + Phase 4))"
else
    echo "Downstream       : NOT auto-chained (ZAC_NO_CHAIN=1 or task != 0)"
    echo "                   re-run with chain enabled:  sbatch scripts/run_array.sbatch"
fi
echo "================================================================="
