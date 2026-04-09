#!/usr/bin/env bash
#SBATCH --job-name=zac_protocol_ab
#SBATCH --partition=gpu_spot
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --constraint=vram24g&cc8.9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/zac_protocol_ab_%j.out
#SBATCH --error=logs/zac_protocol_ab_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# ZeroAutoCL — §10 Protocol Fix A/B test (batch=64 push-the-limit run)
#
# Two runs of the same (encoder L=10/H=64/O=320, GGS_STRATEGY, seed=42) on
# ETTh1 univariate forecasting:
#
#   baseline : crop_len=201,  batch_size=64, legacy per-split eval
#   fixed    : crop_len=3000, batch_size=64, prefix-encoding eval (with val)
#
# This run probes whether a 4090 24 GB card can sustain the §10.1 fixed
# protocol at the original AutoCLS batch_size=64. Previous run at batch=32
# already gave mean MSE 0.155 (see outputs/test_protocol_ab/diff.txt);
# batch=64 doubles the negatives and *may* push performance further, but
# may also OOM. If srun aborts with CUDA OOM, drop the fixed batch_size
# inside scripts/test_protocol_ab.py to 48 (or back to 32) and resubmit.
#
# pretrain_iters is fixed at 600 for BOTH runs so that the only differences
# are §10.1 (crop length) + §10.3 (prefix encoding).
#
# Hardware: 1× RTX 4090 (24 GB), 8 CPUs, 32 GB RAM.
#
# Submit :  sbatch scripts/test_protocol_ab.sbatch
# Watch  :  tail -f logs/zac_protocol_ab_<jobid>.out
# Cancel :  scancel <jobid>
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/ZeroAutoCL}"
cd "$PROJECT_ROOT"

OUT_DIR="outputs/test_protocol_ab"
mkdir -p logs "${OUT_DIR}"

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

# ── GPU sanity check ────────────────────────────────────────────────────────
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| dev:', torch.cuda.get_device_name(0), '| vram_gb:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))"

# Note: NO 50% VRAM cap here. The fixed group uses crop=3000 batch=32
# which is intentionally pushed near the 24 GB ceiling — capping to 12 GB
# would force OOM. If you need to coexist with other jobs, lower batch_size
# inside scripts/test_protocol_ab.py instead of capping VRAM globally.

# ─────────────────────────────────────────────────────────────────────────────
# Run the A/B test
# ─────────────────────────────────────────────────────────────────────────────
echo
echo "═══════ A/B test: baseline (201/64/legacy) vs fixed (3000/32/prefix) ═══════"
srun --unbuffered python -u scripts/test_protocol_ab.py 2>&1 \
    | tee "${OUT_DIR}/run.log"

echo "================================================================="
echo "Finished       : $(date -Is)"
echo "Outputs:"
echo "  baseline   : ${OUT_DIR}/baseline.json"
echo "  fixed      : ${OUT_DIR}/fixed.json"
echo "  diff       : ${OUT_DIR}/diff.txt"
echo "  full log   : ${OUT_DIR}/run.log"
echo "================================================================="
