#!/bin/bash
#SBATCH --job-name=bdh-proto
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/proto_%j.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/proto_%j.out

# Nearest-class-mean readout on exp5's existing checkpoints. No training: this
# re-reads models already on disk, so it costs eval time only.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8 WANDB_MODE=disabled
cd "$WS/code"
echo "=== $(date) job=$SLURM_JOB_ID node=$(hostname) ==="

echo "########## tests ##########"
"$WS/venv/bin/python" -m pytest bdh_cl/tests -q 2>&1 | tail -12
echo "pytest exit: ${PIPESTATUS[0]}"

echo "########## prototype readout on exp5 checkpoints ##########"
"$WS/venv/bin/python" -u bdh_cl/scripts/proto_eval.py \
  --config bdh_cl/configs/exp5_split_mnist.yaml \
  --ckpt-dir bdh_cl/results/exp5_split_mnist/ckpt \
  --models bdh_den_gated,bdh_den_sum,bdh_den_max,bdh,bdh_narrow,gpt \
  --strategies naive,replay,joint \
  --seeds 0,1,2 \
  --out "$WS/results/proto_exp5.json"

echo "=== done $(date) ==="
