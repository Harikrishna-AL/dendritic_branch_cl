#!/bin/bash
#SBATCH --job-name=bdh-ncm
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/ncm_%j.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/ncm_%j.out

# Training-free class-incremental floor: nearest class mean on raw pixels and
# on fixed random sparse (kWTA / dendritic) features. Split MNIST 5x2 and
# Split CIFAR-100 10x10. No learning, so no forgetting -- this is the number a
# continual-learning method must beat to claim its features matter.
set -euo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
PY="$WS/venv/bin/python"
export OMP_NUM_THREADS=8 WANDB_MODE=disabled
cd "$WS/code"
OUT="$WS/results/ncm_baselines"; mkdir -p "$OUT"

"$PY" -m pytest -q bdh_cl/tests/test_ncm_baselines.py
echo "=== smoke ==="
"$PY" bdh_cl/scripts/ncm_baselines.py --dataset cifar100 --cifar-root "$WS/data_cache/cifar100" --seeds 0 --sizes 500 --limit 2000
echo "=== mnist ==="
"$PY" bdh_cl/scripts/ncm_baselines.py --dataset mnist --mnist-cache data_cache/mnist --seeds 0,1,2 --sizes 2000,10000 --out "$OUT/mnist.json"
echo "=== cifar100 ==="
"$PY" bdh_cl/scripts/ncm_baselines.py --dataset cifar100 --cifar-root "$WS/data_cache/cifar100" --seeds 0,1,2 --sizes 2000,10000 --out "$OUT/cifar100.json"
echo "=== done $(date) ==="
