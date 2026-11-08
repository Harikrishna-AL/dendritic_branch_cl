#!/bin/bash
#SBATCH --job-name=bdh-gmm
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/gmm_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/gmm_%A_%a.out
set -euo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=4 PYTHONHASHSEED=0 WANDB_MODE=disabled PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "$WS/code"
"$WS/venv/bin/python" -m pytest bdh_cl/tests/test_gaussian_routing.py -v -s -o faulthandler_timeout=60
srun -N1 -n1 --gres=gpu:1 "$WS/venv/bin/python" -u bdh_cl/scripts/gaussian_routing_mnist.py --seed "${SLURM_ARRAY_TASK_ID:-0}" --steps 600 --out "$WS/results/exp15_gaussian" ${EXTRA_ARGS:-}
