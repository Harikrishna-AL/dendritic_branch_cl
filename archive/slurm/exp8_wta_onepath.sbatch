#!/bin/bash
#SBATCH --job-name=bdh-exp8
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp8_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp8_%A_%a.out
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
MODELS=(bdh_den_max bdh_den_max_wta05 bdh_den_max_wta05x bdh_den_max_wta05y)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=8
export WANDB_DIR="$WS/wandb" WANDB_CACHE_DIR="$WS/.wandb-cache"
mkdir -p "$WS/logs" "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$WS/code"
echo "=== $(date) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID model=$MODEL node=$(hostname) ==="
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  WANDB_MODE=disabled "$WS/venv/bin/python" -m pytest bdh_cl/tests -q 2>&1 | tail -10
  echo "pytest exit: ${PIPESTATUS[0]}"
fi
srun "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
  --config bdh_cl/configs/exp8_wta_onepath.yaml \
  --models "$MODEL" --tag "${TAG:-exp8_wta_onepath}"
echo "=== done $(date) ==="
