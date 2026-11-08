#!/bin/bash
#SBATCH --job-name=bdh-exp13
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp13_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp13_%A_%a.out

# Depth control, on BOTH benchmarks where the dendritic claim lives.
#   task 0      continual learning (synthetic, all four arms in one run)
#   tasks 1-3   Shakespeare at upstream's exact config, seeds 0-2
#
# 2h, not 1h: baseline-width BDH runs at ~14k tok/s so 3000 steps needs ~58 min
# before probes; a 1h limit timed out all three seeds of exp11.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8
export WANDB_DIR="$WS/wandb" WANDB_CACHE_DIR="$WS/.wandb-cache"
mkdir -p "$WS/logs" "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$WS/code"
echo "=== $(date) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID node=$(hostname) ==="

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  echo "########## tests (gating) ##########"
  WANDB_MODE=disabled "$WS/venv/bin/python" -m pytest bdh_cl/tests -q 2>&1 | tail -10
  RC=${PIPESTATUS[0]}; echo "pytest exit: $RC"
  [ "$RC" != "0" ] && { echo "ABORT: tests failed"; exit 1; }

  echo "########## depth control: continual learning ##########"
  srun "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
    --config bdh_cl/configs/exp13_depth_cl.yaml --tag exp13_depth_cl
else
  SEED=$((SLURM_ARRAY_TASK_ID - 1))
  echo "########## depth control: Shakespeare, seed $SEED ##########"
  srun "$WS/venv/bin/python" -u bdh_cl/scripts/sparsity_original.py \
    --model bdh_deep --seed "$SEED" \
    --steps 3000 --batch-size 32 --block-size 512 --probe-every 100 \
    --out "$WS/results/exp3_original" \
    --data-cache "$WS/data_cache/shakespeare.txt" \
    --wandb-project bdh-cl --wandb-group exp13_depth_shakespeare
fi
echo "=== done $(date) ==="
