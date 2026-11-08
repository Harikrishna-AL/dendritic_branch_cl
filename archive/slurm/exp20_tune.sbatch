#!/bin/bash
#SBATCH --job-name=bdh-exp20-tune
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp20tune_%j.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp20tune_%j.out

# Exp20 phase A: tune each arm's one knob on the VALIDATION split, seed 0.
# 10 tasks x 20 epochs. Test split is untouched until phase B.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
PY="$WS/venv/bin/python"
export OMP_NUM_THREADS=8
cd "$WS/code"
OUT="$WS/results/exp20/tune"; mkdir -p "$OUT"
run() { tag=$1; shift
  echo "=== $tag ==="
  "$PY" bdh_cl/scripts/pop_cl_cifar.py --data "$WS/data_cache/cifar100" --eval-split valid --seed 0 \
      --group exp20_tune --tag "$tag" --out "$OUT/$tag.json" "$@" | grep -E "SUMMARY|Error" || echo "FAILED $tag"
}
run finetune        --arm finetune
run finetune_local  --arm finetune_local
run joint           --arm joint
for l in 1 3 10;          do run lwf_l$l   --arm lwf --lwf-lambda $l; done
for l in 10 100 1000 10000; do run ewc_l$l --arm ewc --ewc-lambda $l; done
for k in 50 100 200;      do run pop_k$k   --arm pop --pop-k $k; done
for t in 0.25 1 4;        do run tired_t$t --arm pop_tired --tau-tasks $t; done
echo "=== done $(date) ==="
