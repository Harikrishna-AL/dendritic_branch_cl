#!/bin/bash
#SBATCH --job-name=bdh-exp20-noconv
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp20noconv_%j.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp20noconv_%j.out

# Exp20b: no convolution. Population layers directly on pixels, so every
# learned weight sits inside the branch/top-k/tiredness mechanism. Question:
# does it beat fixed random dendritic features + NCM (0.151 test) and get
# close to its own joint-training ceiling? Validation split, seed 0.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
PY="$WS/venv/bin/python"
export OMP_NUM_THREADS=8
cd "$WS/code"
"$PY" -m pytest -q bdh_cl/tests/test_pop_cl_cifar.py || exit 1
OUT="$WS/results/exp20/noconv"; mkdir -p "$OUT"
run() { tag=$1; shift
  echo "=== $tag ==="
  "$PY" bdh_cl/scripts/pop_cl_cifar.py --data "$WS/data_cache/cifar100" --eval-split valid --seed 0 \
      --group exp20_noconv --tag "$tag" --out "$OUT/$tag.json" "$@" | grep -E "SUMMARY|Error" || echo "FAILED $tag"
}
run joint_mlp        --arm joint     --backbone mlp
run joint_pop_L1     --arm joint_pop --backbone none --pop-layers 1
run joint_pop_L2     --arm joint_pop --backbone none --pop-layers 2
run ftlocal_mlp      --arm finetune_local --backbone mlp
run lwf3_mlp         --arm lwf --lwf-lambda 3 --backbone mlp
run pop_L1           --arm pop --backbone none --pop-layers 1
run pop_L2           --arm pop --backbone none --pop-layers 2
run tired_L1_t0.25   --arm pop_tired --backbone none --pop-layers 1 --tau-tasks 0.25
run tired_L1_t1      --arm pop_tired --backbone none --pop-layers 1 --tau-tasks 1
run tired_L2_t0.25   --arm pop_tired --backbone none --pop-layers 2 --tau-tasks 0.25
echo "=== done $(date) ==="
