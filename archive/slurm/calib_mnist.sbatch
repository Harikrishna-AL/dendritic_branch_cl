#!/bin/bash
#SBATCH --job-name=bdh-calib-mnist
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/calibmnist_%j.out
set -euo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8 WANDB_MODE=disabled
cd "$WS/code"
echo "=== $(date) job=$SLURM_JOB_ID node=$(hostname) ==="
srun "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
  --config bdh_cl/configs/calib_mnist.yaml --tag calib_mnist
echo "=== done $(date) ==="
