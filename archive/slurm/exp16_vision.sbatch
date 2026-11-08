#!/bin/bash
#SBATCH --job-name=bdh-cifar-pre
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/vision_%j.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/vision_%j.out
set -euo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=4 PYTHONHASHSEED=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export WANDB_MODE=online WANDB_DIR="$WS/wandb" WANDB_CACHE_DIR="$WS/.wandb-cache"
export WANDB_INIT_TIMEOUT=60
cd "$WS/code"
PHASE=${PHASE:-pretrain}
if [[ "$PHASE" == smoke ]]; then
  "$WS/venv/bin/python" -m pytest bdh_cl/tests/test_vision_bdh.py -v -s -o faulthandler_timeout=60
  srun -N1 -n1 --gres=gpu:1 "$WS/venv/bin/python" -u bdh_cl/scripts/vision_pretrain_cl.py --phase pretrain --smoke --no-wandb --data "$WS/data_cache/cifar100" --out "$WS/results/exp16_smoke/base"
  srun -N1 -n1 --gres=gpu:1 "$WS/venv/bin/python" -u bdh_cl/scripts/vision_pretrain_cl.py --phase adapt --smoke --no-wandb --data "$WS/data_cache/cifar100" --pretrained "$WS/results/exp16_smoke/base" --out "$WS/results/exp16_smoke/adapt"
elif [[ "$PHASE" == pretrain ]]; then
  srun -N1 -n1 --gres=gpu:1 "$WS/venv/bin/python" -u bdh_cl/scripts/vision_pretrain_cl.py --phase pretrain --data "$WS/data_cache/cifar100" --out "$WS/results/exp16_cifar100/base_seed0"
else
  srun -N1 -n1 --gres=gpu:1 "$WS/venv/bin/python" -u bdh_cl/scripts/vision_pretrain_cl.py --phase adapt --data "$WS/data_cache/cifar100" --pretrained "$WS/results/exp16_cifar100/base_seed0" --out "$WS/results/exp16_cifar100/adapt_seed0"
fi
