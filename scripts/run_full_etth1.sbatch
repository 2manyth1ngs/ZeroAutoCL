#!/usr/bin/env bash
#SBATCH --job-name=zac_etth1
#SBATCH --partition=gpu_spot
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --constraint=vram24g&cc8.9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/zac_etth1_%j.out
#SBATCH --error=logs/zac_etth1_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# ZeroAutoCL — paper-level forecasting experiment on ETTh1
#
#   Target  : ETTh1
#   Sources : ETTm1, PEMS03, PEMS04, PEMS07, PEMS08, ExchangeRate, PEMS-BAY
#             (every forecasting dataset except the ETTh family — exactly the
#             default --datasets list inside run_generate_seeds.py)
#
# Pipeline (each phase is a dedicated CLI script driven by configs/default.yaml,
# which already encodes the paper-level budgets — n_per_dataset=200,
# pretrain_iters=600, comparator epochs=250, candidate pool=200k, top_k=30,
# tournament_rounds=25):
#
#   Phase 1   : scripts/run_generate_seeds.py        → outputs/full_etth1/seeds/seeds.json
#   Phase 1.5 : scripts/run_pretrain_comparator.py   → outputs/full_etth1/comparator.pt
#   Phase 2+3 : scripts/run_zero_shot_search.py      → outputs/full_etth1/best_ETTh1.json
#   Phase 4   : inline retrain + test eval           → outputs/full_etth1/test_eval.json
#
# Hardware:
#   * 1 x RTX 4090 (gpu_spot, up to 72h, preemptible + auto-requeue)
#   * 8 CPU threads, 32 GB RAM
#   * Physical VRAM = 24 GB; PyTorch capped to ~12 GB at runtime via
#     torch.cuda.set_per_process_memory_fraction(0.5).
#
# Submit :  sbatch scripts/run_full_etth1.sbatch
# Watch  :  squeue -u $USER
#           tail -f logs/zac_etth1_<jobid>.out
#           sstat --format=JobID,TRESUsageInTot%115 -j <jobid> --allsteps
# Cancel :  scancel <jobid>
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Project root (edit if your checkout lives elsewhere) ────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/ZeroAutoCL}"
cd "$PROJECT_ROOT"

OUT_DIR="outputs/full_etth1"
SEED_DIR="${OUT_DIR}/seeds"
SEEDS_JSON="${SEED_DIR}/seeds.json"
COMPARATOR_PT="${OUT_DIR}/comparator.pt"
BEST_JSON="${OUT_DIR}/best_ETTh1.json"
TEST_JSON="${OUT_DIR}/test_eval.json"
CONFIG="configs/default.yaml"
DATA_DIR="data/datasets"

mkdir -p logs "${SEED_DIR}"

echo "================================================================="
echo "Job ID         : ${SLURM_JOB_ID}"
echo "Node           : $(hostname)"
echo "Partition      : ${SLURM_JOB_PARTITION}"
echo "GPUs visible   : ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Project root   : ${PROJECT_ROOT}"
echo "Started        : $(date -Is)"
echo "================================================================="

# ── Software environment ────────────────────────────────────────────────────

module load anaconda

conda activate zeroautocl

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ── GPU sanity check + 12 GB VRAM cap announcement ─────────────────────────
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| dev:', torch.cuda.get_device_name(0))"

# A reusable PYTHONSTARTUP file that caps PyTorch VRAM to 50% of the 24 GB
# 4090 (≈12 GB) for every Python process spawned by this job.  We write it
# once into a temp file and export PYTHONSTARTUP so all four phases inherit.
VRAM_CAP_INIT="$(mktemp --suffix=.py)"
cat >"${VRAM_CAP_INIT}" <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5, device=0)
        print("[vram-cap] PyTorch VRAM cap = 50% of",
              torch.cuda.get_device_name(0), "(~12 GB)", flush=True)
except Exception as _e:
    print("[vram-cap] failed:", _e, flush=True)
PY
export PYTHONSTARTUP="${VRAM_CAP_INIT}"
trap 'rm -f "${VRAM_CAP_INIT}"' EXIT

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — generate seeds on the 7 source datasets
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Phase 1: seed generation ═══════"
if [ -f "${SEEDS_JSON}" ]; then
    echo "[skip] ${SEEDS_JSON} already exists — Phase 1 skipped (resume mode)."
else
    srun --unbuffered python -u scripts/run_generate_seeds.py \
        --data_dir "${DATA_DIR}" \
        --save_dir "${SEED_DIR}" \
        --config   "${CONFIG}" \
        --seed     42
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1.5 — pretrain T-CLSC comparator on those seeds
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Phase 1.5: comparator pretraining ═══════"
if [ -f "${COMPARATOR_PT}" ]; then
    echo "[skip] ${COMPARATOR_PT} already exists — Phase 1.5 skipped (resume mode)."
else
    srun --unbuffered python -u scripts/run_pretrain_comparator.py \
        --seeds_path "${SEEDS_JSON}" \
        --data_dir   "${DATA_DIR}" \
        --save_path  "${COMPARATOR_PT}" \
        --config     "${CONFIG}" \
        --seed       42
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 + 3 — zero-shot search + top-K full retrain on ETTh1
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Phase 2+3: zero-shot search on ETTh1 ═══════"
if [ -f "${BEST_JSON}" ]; then
    echo "[skip] ${BEST_JSON} already exists — Phase 2+3 skipped (resume mode)."
else
    srun --unbuffered python -u scripts/run_zero_shot_search.py \
        --comparator_path "${COMPARATOR_PT}" \
        --data_dir        "${DATA_DIR}" \
        --target_dataset  ETTh1 \
        --config          "${CONFIG}" \
        --output          "${BEST_JSON}" \
        --seed            42
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — final retrain of the winning config + ETTh1 test-set evaluation
# (the three CLI scripts above stop at val score; this last step gives the
# multi-horizon test MSE/MAE that the paper reports)
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ Phase 4: ETTh1 test-set evaluation ═══════"
srun --unbuffered python -u - <<PYEOF
import json, sys, torch, os
sys.path.insert(0, ".")

from data.dataset import load_dataset
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.contrastive.cl_pipeline import CLPipeline
from train.pretrain import contrastive_pretrain
from train.evaluate import eval_forecasting
from utils.reproducibility import set_seed

BEST_JSON = "${BEST_JSON}"
TEST_JSON = "${TEST_JSON}"
DATA_DIR  = "${DATA_DIR}"
TARGET    = "ETTh1"
ITERS     = 600    # matches dataset_budgets[ETTh1].pretrain_iters in default.yaml

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(BEST_JSON) as f:
    best = json.load(f)
best_enc   = best["best_encoder_config"]
best_strat = best["best_strategy"]
print(f"[phase4] best encoder = {best_enc}")
print(f"[phase4] best strategy = {best_strat}")

splits = load_dataset(TARGET, DATA_DIR)
train_ds, test_ds = splits["train"], splits["test"]

encoder  = DilatedCNNEncoder.from_config_dict(train_ds.n_channels, best_enc).to(device)
pipeline = CLPipeline(encoder, best_strat).to(device)
contrastive_pretrain(
    encoder=encoder,
    cl_pipeline=pipeline,
    train_data=train_ds,
    config={"pretrain_iters": ITERS, "pretrain_lr": 1e-3, "batch_size": 64},
    device=device,
    task_type="forecasting",
)
encoder.eval()
metrics = eval_forecasting(
    encoder, train_ds, test_ds,
    horizons=[24, 48, 168, 336, 720], device=device,
)
mean_mse = sum(v["mse"] for v in metrics.values()) / len(metrics)
mean_mae = sum(v["mae"] for v in metrics.values()) / len(metrics)

result = {
    "target": TARGET,
    "best_encoder_config": best_enc,
    "best_strategy": best_strat,
    "val_score": best.get("performance"),
    "test_mean_mse": mean_mse,
    "test_mean_mae": mean_mae,
    "test_per_horizon": metrics,
}
os.makedirs(os.path.dirname(TEST_JSON) or ".", exist_ok=True)
with open(TEST_JSON, "w") as f:
    json.dump(result, f, indent=2)

print("=" * 60)
print(f"FINAL: ETTh1 test mean MSE = {mean_mse:.4f}")
print(f"       ETTh1 test mean MAE = {mean_mae:.4f}")
print(f"Saved -> {TEST_JSON}")
print("=" * 60)
PYEOF

echo "================================================================="
echo "Finished       : $(date -Is)"
echo "Outputs:"
echo "  seeds         : ${SEEDS_JSON}"
echo "  comparator    : ${COMPARATOR_PT}"
echo "  best config   : ${BEST_JSON}"
echo "  test eval     : ${TEST_JSON}"
echo "================================================================="
