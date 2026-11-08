#!/bin/bash
#SBATCH --job-name=bdh-exp21
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp21_%j.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp21_%j.out

# Exp21: untrained population / ReLU expansion + analytic class-incremental
# readouts (NCM, streaming LDA, ridge) on frozen ImageNet ResNet-18 features.
# Shrinkage chosen on valid; test reported over 3 seeds.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
PY="$WS/venv/bin/python"
export OMP_NUM_THREADS=8
cd "$WS/code"
"$PY" -m pytest -q bdh_cl/tests/test_analytic_readout.py || exit 1
FEATS="$WS/data_cache/cifar100_r18_feats.pt"
OUT="$WS/results/exp21"; mkdir -p "$OUT"
echo "=== smoke ==="
"$PY" bdh_cl/scripts/analytic_readout.py --feats "$FEATS" --maps raw,relu_200,pop_200_k10_K4 --seeds 0 --n-tasks 2 || exit 1
echo "=== full ==="
"$PY" bdh_cl/scripts/analytic_readout.py --feats "$FEATS" --seeds 0,1,2 \
  --maps raw,relu_2000,relu_8000,pop_2000_k100_K4,pop_8000_k160_K4,pop_8000_k160_K1,pop_8000_k400_K4 \
  --out "$OUT/exp21.json"
echo "=== done $(date) ==="
