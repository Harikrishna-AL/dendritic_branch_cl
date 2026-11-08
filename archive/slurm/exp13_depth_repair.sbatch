#!/bin/bash
#SBATCH --job-name=bdh-depth-check
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/depth_repair_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/depth_repair_%A_%a.out
set -euo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8
export WANDB_MODE=disabled
cd "$WS/code"
"$WS/venv/bin/python" -m pytest bdh_cl/tests/test_neuron_units.py bdh_cl/tests/test_bdh_equivalence.py bdh_cl/tests/test_cl_integration.py -q
srun "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
    --config bdh_cl/configs/exp13_depth_cl.yaml \
    --seeds "$SLURM_ARRAY_TASK_ID" \
    --tag "exp13_depth_repair_seed${SLURM_ARRAY_TASK_ID}"
