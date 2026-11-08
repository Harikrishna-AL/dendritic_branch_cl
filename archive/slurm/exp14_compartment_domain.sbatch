#!/bin/bash
#SBATCH --job-name=bdh-comp-domain
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/comp_domain_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/comp_domain_%A_%a.out
set -euo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8 PYTHONHASHSEED=0 WANDB_MODE=disabled
cd "$WS/code"
"$WS/venv/bin/python" -m pytest bdh_cl/tests/test_compartment_bdh.py -q
srun "$WS/venv/bin/python" -u bdh_cl/scripts/compartment_recurrence_domain.py --device cuda --seed "$SLURM_ARRAY_TASK_ID" --steps 800 --domain-swap --out "$WS/results/exp14_compartment_domain"
