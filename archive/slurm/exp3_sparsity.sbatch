#!/bin/bash
#SBATCH --job-name=bdh-exp3
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp3_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp3_%A_%a.out

# Sparsity at the ORIGINAL upstream config, BDH vs parameter-matched GPT.
# One array task per model, so they run concurrently on separate GPUs and
# neither can be blamed for the other's throughput.
set -euo pipefail

WS=/data/horse/ws/hapi039h-bdh-d
MODELS=(bdh gpt)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

mkdir -p "$WS/logs"
cd "$WS/code"

echo "=== $(date) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID model=$MODEL node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# W&B credentials come from ~/.netrc, already present on this account.
export WANDB_DIR="$WS/wandb"
export WANDB_CACHE_DIR="$WS/.wandb-cache"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

srun "$WS/venv/bin/python" -u bdh_cl/scripts/sparsity_original.py \
  --model "$MODEL" \
  --steps "${STEPS:-3000}" \
  --batch-size "${BS:-32}" \
  --block-size "${BLK:-512}" \
  --probe-every "${PROBE_EVERY:-100}" \
  --seed "${SEED:-0}" \
  --out "$WS/results/exp3_original" \
  --data-cache "$WS/data_cache/shakespeare.txt" \
  --wandb-project "${WANDB_PROJECT:-bdh-cl}" \
  --wandb-group exp3_original_config

echo "=== done $(date) ==="
