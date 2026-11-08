#!/bin/bash
#SBATCH --job-name=bdh-exp20-wide
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp20wide_%j.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp20wide_%j.out

# Exp20c: does a WIDE population give tiredness fresh capacity? No conv,
# validation split, seed 0. Untrained floors (--epochs 0) use the same layer,
# preprocessing and prototype readout, so the comparison is exact.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
PY="$WS/venv/bin/python"
export OMP_NUM_THREADS=8
cd "$WS/code"
OUT="$WS/results/exp20/wide"; mkdir -p "$OUT"
run() { tag=$1; shift
  echo "=== $tag ==="
  "$PY" bdh_cl/scripts/pop_cl_cifar.py --data "$WS/data_cache/cifar100" --eval-split valid --seed 0 \
      --backbone none --group exp20_wide --tag "$tag" --out "$OUT/$tag.json" "$@" | grep -E "SUMMARY|Error" || echo "FAILED $tag"
}
# untrained floors
run rand_N2000_k100   --arm pop --epochs 0 --pop-n 2000  --pop-k 100
run rand_N10000_k200  --arm pop --epochs 0 --pop-n 10000 --pop-k 200
# sparsity control at the old width
run tired_N2000_k40_t0.25   --arm pop_tired --pop-n 2000  --pop-k 40  --tau-tasks 0.25
# wide
run pop_N8000_k160          --arm pop       --pop-n 8000  --pop-k 160
run tired_N8000_k160_t0.25  --arm pop_tired --pop-n 8000  --pop-k 160 --tau-tasks 0.25
run tired_N8000_k160_t1     --arm pop_tired --pop-n 8000  --pop-k 160 --tau-tasks 1
run tired_N10000_k200_t0.25 --arm pop_tired --pop-n 10000 --pop-k 200 --tau-tasks 0.25
run tired_N10000_k100_t0.25 --arm pop_tired --pop-n 10000 --pop-k 100 --tau-tasks 0.25
echo "=== done $(date) ==="
