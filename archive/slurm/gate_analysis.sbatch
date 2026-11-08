#!/bin/bash
#SBATCH --job-name=bdh-gate
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/gate_%j.out

# Does the trained `gated` soma route, or has it collapsed to the mean?
# Evaluation only -- no training. See bdh_cl/scripts/gate_analysis.py.
set -euo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8
export WANDB_MODE=disabled
cd "$WS/code"

echo "=== $(date) job=$SLURM_JOB_ID node=$(hostname) ==="
srun "$WS/venv/bin/python" -u bdh_cl/scripts/gate_analysis.py \
  --config bdh_cl/configs/exp4_dendritic.yaml \
  --ckpt-dir bdh_cl/results/exp4_dendritic/ckpt \
  --strategy replay --after 2_sort --seeds 0,1,2 \
  --out "$WS/results/gate_analysis_replay.json"

echo "--- and the naive-trained gated models, for contrast ---"
srun "$WS/venv/bin/python" -u bdh_cl/scripts/gate_analysis.py \
  --config bdh_cl/configs/exp4_dendritic.yaml \
  --ckpt-dir bdh_cl/results/exp4_dendritic/ckpt \
  --strategy naive --after 2_sort --seeds 0,1,2 \
  --out "$WS/results/gate_analysis_naive.json"
echo "=== done $(date) ==="
