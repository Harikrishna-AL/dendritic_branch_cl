#!/bin/bash
#SBATCH --job-name=bdh-exp13cl
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp13cl_%j.out
# Depth control on the continual-learning benchmark. Unlike Shakespeare, this
# samples fresh data every step, so it is not over-epoched and held-out
# performance is not dominated by memorisation (r = -0.976 on Shakespeare).
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8 WANDB_DIR="$WS/wandb" WANDB_CACHE_DIR="$WS/.wandb-cache"
cd "$WS/code"
WANDB_MODE=disabled "$WS/venv/bin/python" -m pytest bdh_cl/tests/test_neuron_units.py -q 2>&1 | tail -4
RC=${PIPESTATUS[0]}; echo "pytest exit: $RC"; [ "$RC" != "0" ] && { echo ABORT; exit 1; }
srun "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py --config bdh_cl/configs/exp13_depth_cl.yaml --tag exp13_depth_cl
echo "=== done $(date) ==="
