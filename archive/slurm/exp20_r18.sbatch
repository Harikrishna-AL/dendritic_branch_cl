#!/bin/bash
#SBATCH --job-name=bdh-exp20-r18
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp20r18_%j.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp20r18_%j.out

# Exp20d: population + tiredness on FROZEN ImageNet ResNet-18 features.
# Bar: NCM directly on the frozen features. Validation split, seed 0.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
PY="$WS/venv/bin/python"
export OMP_NUM_THREADS=8 TORCH_HOME="$WS/data_cache/torch_hub"
cd "$WS/code"
"$PY" -m pytest -q bdh_cl/tests/test_pop_cl_cifar.py bdh_cl/tests/test_extract_feats.py || exit 1
FEATS="$WS/data_cache/cifar100_r18_feats.pt"
echo "=== smoke extract ==="
"$PY" bdh_cl/scripts/extract_feats.py --data "$WS/data_cache/cifar100" --out "$WS/data_cache/smoke_feats.pt" --limit 64 || exit 1
if [ ! -f "$FEATS" ]; then
  echo "=== extract ==="
  "$PY" bdh_cl/scripts/extract_feats.py --data "$WS/data_cache/cifar100" --out "$FEATS" || exit 1
fi
OUT="$WS/results/exp20/r18"; mkdir -p "$OUT"
run() { tag=$1; shift
  echo "=== $tag ==="
  "$PY" bdh_cl/scripts/pop_cl_cifar.py --data "$WS/data_cache/cifar100" --feats "$FEATS" --eval-split valid --seed 0 \
      --group exp20_r18 --tag "$tag" --out "$OUT/$tag.json" "$@" | grep -E "SUMMARY|Error" || echo "FAILED $tag"
}
run ncm_feats            --arm finetune_local --backbone none --epochs 0
run joint_linear         --arm joint          --backbone none
run rand_N2000_k100      --arm pop  --backbone none --epochs 0 --pop-n 2000 --pop-k 100
run rand_N8000_k160      --arm pop  --backbone none --epochs 0 --pop-n 8000 --pop-k 160
run joint_pop_N2000      --arm joint_pop --backbone none --pop-n 2000 --pop-k 100
run ftlocal_mlp          --arm finetune_local --backbone mlp
run lwf3_mlp_clip        --arm lwf --lwf-lambda 3 --backbone mlp --clip 1.0
run pop_N2000_k100       --arm pop       --backbone none --pop-n 2000 --pop-k 100
run tired_N2000_t0.25    --arm pop_tired --backbone none --pop-n 2000 --pop-k 100 --tau-tasks 0.25
run tired_N2000_t1       --arm pop_tired --backbone none --pop-n 2000 --pop-k 100 --tau-tasks 1
run tired_N8000_t0.25    --arm pop_tired --backbone none --pop-n 8000 --pop-k 160 --tau-tasks 0.25
echo "=== done $(date) ==="
