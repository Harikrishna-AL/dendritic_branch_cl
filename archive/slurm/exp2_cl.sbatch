#!/bin/bash
#SBATCH --job-name=bdh-exp2
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp2_%j.out

# Step 3 continual-learning sweep, moved off the laptop onto a GPU.
# Runs the FULL sweep (all seeds, both models, all strategies) from scratch
# rather than resuming, so every number in the report comes from one
# consistent environment.
set -euo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8
export WANDB_DIR="$WS/wandb"
export WANDB_CACHE_DIR="$WS/.wandb-cache"
mkdir -p "$WS/logs" "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$WS/code"

echo "=== $(date) job=$SLURM_JOB_ID node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
  --config "bdh_cl/configs/${CONFIG:-exp2_synthetic3}.yaml" \
  --tag "${TAG:-exp2_synthetic3_gpu}"

echo "=== done $(date) ==="
