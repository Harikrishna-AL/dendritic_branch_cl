#!/bin/bash
#SBATCH --job-name=bdh-exp20-smoke
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp20smoke_%j.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp20smoke_%j.out

# Exp20 smoke: unit tests, then every arm for 2 tasks x 1 epoch on full
# per-class data to measure seconds/epoch before sizing the real sweep.
set -euo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
PY="$WS/venv/bin/python"
export OMP_NUM_THREADS=8 WANDB_MODE=disabled
cd "$WS/code"
"$PY" -m pytest -q bdh_cl/tests/test_pop_cl_cifar.py
for arm in finetune finetune_local lwf ewc pop pop_tired joint; do
  echo "=== $arm ==="
  "$PY" bdh_cl/scripts/pop_cl_cifar.py --data "$WS/data_cache/cifar100" --arm $arm --epochs 1 --n-tasks 2 --no-wandb
done
echo "=== done $(date) ==="
