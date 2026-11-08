#!/bin/bash
#SBATCH --job-name=bdh-exp5
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp5_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp5_%A_%a.out

# Split MNIST, class-incremental. One array task per model so they run
# concurrently on separate GPUs and a single model's failure does not take the
# sweep with it. All six write per-seed JSON into the same results dir; the
# seed aggregation and plots are produced afterwards by report_tables.py.
set -euo pipefail

WS=/data/horse/ws/hapi039h-bdh-d
MODELS=(bdh gpt bdh_narrow bdh_den_gated bdh_den_sum bdh_den_max)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

export OMP_NUM_THREADS=8
export WANDB_DIR="$WS/wandb"
export WANDB_CACHE_DIR="$WS/.wandb-cache"
mkdir -p "$WS/logs" "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$WS/code"

echo "=== $(date) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID model=$MODEL node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
  --config bdh_cl/configs/exp5_split_mnist.yaml \
  --models "$MODEL" \
  --tag "${TAG:-exp5_split_mnist}"

echo "=== done $(date) ==="
