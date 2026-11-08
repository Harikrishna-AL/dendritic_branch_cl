#!/bin/bash
#SBATCH --job-name=bdh-interf
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp17_%j.out
# Cross-task interference probe. Same models, preset, tasks and training spec
# as exp13 (CL depth control), so the numbers line up with its retention
# results: dendritic 0.948, baseline 0.830, depth control 0.787.
# Measures how much a burst of B updates damages A, per unit of B learned,
# and how localised the change to A's code is. ~15 min of compute.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8 WANDB_MODE=disabled
cd "$WS/code"
echo "=== $(date) job=$SLURM_JOB_ID node=$(hostname) ==="
"$WS/venv/bin/python" -m pytest bdh_cl/tests -q 2>&1 | tail -6
RC=${PIPESTATUS[0]}; echo "pytest exit: $RC"; [ "$RC" != "0" ] && { echo "ABORT: tests failed"; exit 1; }
srun "$WS/venv/bin/python" -u bdh_cl/scripts/interference_probe.py \
  --config bdh_cl/configs/exp13_depth_cl.yaml \
  --models bdh,bdh_den_sum,bdh_den_gated,bdh_deep,gpt \
  --pairs copy:reverse,reverse:sort --seeds 0,1,2 --b-steps 1,5,20,50 \
  --out "$WS/results/exp17_interference.json"
echo "=== done $(date) ==="
