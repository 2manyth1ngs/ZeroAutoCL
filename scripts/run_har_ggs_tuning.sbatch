#!/usr/bin/env bash
#SBATCH --job-name=har_ggs_tune
#SBATCH --partition=gpu_spot
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --gpu-bind=closest
#SBATCH --exclusive
#SBATCH --constraint=vram24g&cc8.9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/har_ggs_tune_%j.out
#SBATCH --error=logs/har_ggs_tune_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# HAR GGS parameter tuning — 5 configurations, ~10 min total on 4090.
#
# Submit  :  sbatch scripts/run_har_ggs_tuning.sbatch
# Watch   :  tail -f logs/har_ggs_tune_<jobid>.out
# Cancel  :  scancel <jobid>
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/ZeroAutoCL}"
cd "$PROJECT_ROOT"

SEED=42

mkdir -p logs outputs/har_ggs_tuning

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
export PYTHONHASHSEED="${SEED}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| dev:', torch.cuda.get_device_name(0))"

# ─────────────────────────────────────────────────────────────────────────────
srun --unbuffered python -u scripts/test_har_ggs_tuning.py \
    > >(tee -a outputs/har_ggs_tuning/run.log) \
    2> >(tee -a outputs/har_ggs_tuning/run.log >&2)

echo "================================================================="
echo "Finished       : $(date -Is)"
echo "Log            : outputs/har_ggs_tuning/run.log"
echo "================================================================="
