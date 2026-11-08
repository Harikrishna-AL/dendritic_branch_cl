#!/bin/bash
#SBATCH --job-name=bdh-exp19
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp19_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp19_%A_%a.out
# Step 3: swap SET rewiring (exp18) for Hebbian/coactivity-driven regrowth,
# same 2x2 shape (exp19_coactivity_rewire.yaml). 12 runs + references per seed
# at ~100 s each on CUDA (exp5/exp18 timing) => ~25 min per array task.
set -euo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8
export WANDB_DIR="$WS/wandb" WANDB_CACHE_DIR="$WS/.wandb-cache" WANDB_INIT_TIMEOUT=60
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$WS/code"
WANDB_MODE=disabled "$WS/venv/bin/python" -m pytest bdh_cl/tests/test_neuron_units.py bdh_cl/tests/test_bdh_equivalence.py bdh_cl/tests/test_cl_integration.py bdh_cl/tests/test_sparse_dendritic.py -q
export WANDB_MODE=online
srun "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
    --config bdh_cl/configs/exp19_coactivity_rewire.yaml \
    --seeds "$SLURM_ARRAY_TASK_ID" \
    --tag "exp19_coactivity_rewire_seed${SLURM_ARRAY_TASK_ID}"
