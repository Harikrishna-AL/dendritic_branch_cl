#!/bin/bash
#SBATCH --job-name=bdh-comp-pilot
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/comp_pilot_%j.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/comp_pilot_%j.out
set -euo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8 PYTHONHASHSEED=0 WANDB_MODE=disabled
cd "$WS/code"
"$WS/venv/bin/python" -m pytest bdh_cl/tests/test_compartment_bdh.py -q
srun "$WS/venv/bin/python" -u bdh_cl/scripts/compartment_recurrence.py --device cuda --seed 0 --steps 300 --out "$WS/results/exp14_compartment_pilot"
