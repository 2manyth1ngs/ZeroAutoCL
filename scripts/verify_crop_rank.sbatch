#!/usr/bin/env bash
#SBATCH --job-name=zac_verify_crop
#SBATCH --partition=gpu_spot
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --gpu-bind=closest
#SBATCH --constraint=vram24g&cc8.9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/zac_verify_crop_%j.out
#SBATCH --error=logs/zac_verify_crop_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# ZeroAutoCL — rank-preservation check for seed_generation.crop_len.
#
# 5 CL strategies (spanning short→long context) × 2 crop sizes (1024, 3000)
# = 10 pretrain+eval runs on ETTh1, ~10 min each on a 4090 → ~100 min total.
# Encoder is fixed at GGS L10/H64/O320 to isolate the CL strategy variable.
#
# Output: outputs/crop_rank_check/crop_rank_results.json
# Stdout : the script prints a final "Spearman ρ (MSE) = X.XXX" verdict.
#
#   ρ ≥ 0.8  → keep crop_len=1024
#   0.6–0.8 → consider 2048
#   ρ < 0.6 → raise to ≥2048 in seed_generation
#
# Submit:  sbatch scripts/verify_crop_rank.sbatch
# Watch :  tail -f logs/zac_verify_crop_<jobid>.out
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/ZeroAutoCL}"
cd "$PROJECT_ROOT"

OUT_DIR="outputs/crop_rank_check"
DATA_DIR="data/datasets"
TARGET="ETTh1"
SEED=42

mkdir -p logs "${OUT_DIR}"

echo "================================================================="
echo "Job ID         : ${SLURM_JOB_ID:-unknown}"
echo "Node           : $(hostname)"
echo "Partition      : ${SLURM_JOB_PARTITION:-unknown}"
echo "GPUs visible   : ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Project root   : ${PROJECT_ROOT}"
echo "Output dir     : ${OUT_DIR}"
echo "Target         : ${TARGET}"
echo "Started        : $(date -Is)"
echo "================================================================="

module load anaconda
conda activate zeroautocl

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${SEED}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| dev:', torch.cuda.get_device_name(0))"

echo
echo "═══════ verify_crop_rank: 5 candidates × 2 crops on ${TARGET} ═══════"
srun --unbuffered python -u scripts/verify_crop_rank.py \
    --target   "${TARGET}" \
    --data_dir "${DATA_DIR}" \
    --save_dir "${OUT_DIR}" \
    --seed     "${SEED}" \
    --iters    600 \
    --lr       0.001 \
    --batch_size 32 \
    > >(tee -a "${OUT_DIR}/run.log") \
    2> >(tee -a "${OUT_DIR}/run.log" >&2)

echo
echo "================================================================="
echo "Finished       : $(date -Is)"
echo "Results JSON   : ${OUT_DIR}/crop_rank_results.json"
echo "Log            : ${OUT_DIR}/run.log"
echo "================================================================="

# Print final verdict from the JSON (in case stdout was cut off mid-run).
if [[ -s "${OUT_DIR}/crop_rank_results.json" ]]; then
    echo
    echo ">>> Verdict:"
    python -c "
import json
with open('${OUT_DIR}/crop_rank_results.json') as f:
    s = json.load(f)
rho_mse = s.get('spearman_mse')
rho_mae = s.get('spearman_mae')
print(f'  Spearman ρ (MSE) : {rho_mse:.3f}' if rho_mse is not None else '  Spearman ρ (MSE) : n/a')
print(f'  Spearman ρ (MAE) : {rho_mae:.3f}' if rho_mae is not None else '  Spearman ρ (MAE) : n/a')
print()
for c, ranking in s.get('ranked_by_mse', {}).items():
    print(f'  Ranking @ crop={c}:')
    for i, (name, mse) in enumerate(ranking, 1):
        print(f'    {i}. {name:20s}  mean_mse={mse:.4f}')
print()
if rho_mse is not None:
    if rho_mse >= 0.8:
        print('  >>> GOOD: keep crop_len=1024')
    elif rho_mse >= 0.6:
        print('  >>> MILD BIAS: consider crop_len=2048')
    else:
        print('  >>> STRONG BIAS: raise crop_len to >=2048')
"
fi
