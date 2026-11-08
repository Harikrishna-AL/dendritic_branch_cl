#!/bin/bash
#SBATCH --job-name=bdh-exp10
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# 1h, not 4h: the measured run is 35:41, and on a drained cluster a 4-hour
# request cannot be backfilled into the gaps a 1-hour one fits into. The
# original 4h ask left five tasks pending for a day.
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp10_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp10_%A_%a.out

# Dendritic BDH on tiny Shakespeare at UPSTREAM'S EXACT CONFIG, to compare
# against the baseline numbers already measured with this same script:
#
#   BDH  train bpb 0.1605 +- 0.0006   held-out 4.575 +- 0.031
#   GPT  train bpb 1.073  +- 0.108    held-out 2.640 +- 0.126
#
# Held identical to upstream: 6 layers, d=256, 4 heads, dropout 0.1, block 512,
# batch 32, 3000 steps, AdamW 1e-3 / wd 0.1, no schedule, no clipping, fp32,
# same corpus and 90/10 split. The ONLY changes are the neuron unit and
# mlp_internal_dim_multiplier 128 -> 43, which has to move because the two
# encoders scale with k; holding it fixed would give the dendritic model ~3x
# the parameters and measure capacity instead of architecture. 43 puts the
# total at 25.49M against upstream's 25.30M (+0.78%).
#
# `gated` is absent deliberately: it materialises the branch softmax alongside
# the branch activations and OOMs at 93.1 GiB (measured, job 4320011). sum and
# max peak at 70.2 and 76.1 GiB. Running gated would require a batch-size
# deviation, so it is left out rather than silently changing the config.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
MODELS=(bdh_den_sum bdh_den_sum bdh_den_sum bdh_den_max bdh_den_max bdh_den_max)
SEEDS=(0 1 2 0 1 2)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

export OMP_NUM_THREADS=8
export WANDB_DIR="$WS/wandb" WANDB_CACHE_DIR="$WS/.wandb-cache"
mkdir -p "$WS/logs" "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$WS/code"
echo "=== $(date) job=$SLURM_JOB_ID model=$MODEL seed=$SEED node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun "$WS/venv/bin/python" -u bdh_cl/scripts/sparsity_original.py \
  --model "$MODEL" --seed "$SEED" \
  --steps 3000 --batch-size 32 --block-size 512 --probe-every 100 \
  --out "$WS/results/exp3_original" \
  --data-cache "$WS/data_cache/shakespeare.txt" \
  --wandb-project bdh-cl --wandb-group exp10_dendritic_shakespeare

echo "=== done $(date) ==="
