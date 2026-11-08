#!/bin/bash
#SBATCH --job-name=bdh-exp7
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp7_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp7_%A_%a.out

# Winner-take-all variants, one array task per model. Array task 0 also runs
# the test suite and the smoke config first, so a wiring problem surfaces
# before five other GPUs have spent an hour on it.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
MODELS=(bdh_den_gated bdh_den_max bdh_den_topk2 bdh_den_max_wta10 bdh_den_max_wta05 bdh_den_sum_wta05)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

export OMP_NUM_THREADS=8
export WANDB_DIR="$WS/wandb" WANDB_CACHE_DIR="$WS/.wandb-cache"
mkdir -p "$WS/logs" "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$WS/code"
echo "=== $(date) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID model=$MODEL node=$(hostname) ==="

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  echo "########## tests ##########"
  WANDB_MODE=disabled "$WS/venv/bin/python" -m pytest bdh_cl/tests -q 2>&1 | tail -12
  echo "pytest exit: ${PIPESTATUS[0]}"
  echo "########## smoke ##########"
  WANDB_MODE=disabled "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
    --config bdh_cl/configs/smoke_wta.yaml --tag smoke_wta 2>&1 | tail -20
fi

echo "########## exp7: $MODEL ##########"
srun "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
  --config bdh_cl/configs/exp7_wta.yaml \
  --models "$MODEL" --tag "${TAG:-exp7_wta}"

echo "=== done $(date) ==="
