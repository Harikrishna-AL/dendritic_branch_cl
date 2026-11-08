#!/bin/bash
#SBATCH --job-name=bdh-exp18
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp18_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp18_%A_%a.out
# Step 2: branch x rewiring 2x2 (exp18_sparse_rewire.yaml). 12 runs + references
# per seed at ~100 s each on CUDA (exp5 timing) => ~25 min per array task.
set -euo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8
export WANDB_MODE=disabled
cd "$WS/code"
"$WS/venv/bin/python" -m pytest bdh_cl/tests/test_neuron_units.py bdh_cl/tests/test_bdh_equivalence.py bdh_cl/tests/test_cl_integration.py bdh_cl/tests/test_sparse_dendritic.py -q
srun "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
    --config bdh_cl/configs/exp18_sparse_rewire.yaml \
    --seeds "$SLURM_ARRAY_TASK_ID" \
    --tag "exp18_sparse_rewire_seed${SLURM_ARRAY_TASK_ID}"
