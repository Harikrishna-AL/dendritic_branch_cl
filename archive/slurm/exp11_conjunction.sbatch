#!/bin/bash
#SBATCH --job-name=bdh-exp11
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# 2h. The 1h first attempt timed out all three seeds: it was sized from exp10's
# 36-minute runs, but those are the DENDRITIC model at 24k tok/s. Baseline BDH
# runs at 14k tok/s, so 3000 steps x 16384 tokens needs ~58 min of training
# alone, before the test gate and the probes. The original exp3 script used 2h
# for this same model, which is the number to have trusted.
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp11_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp11_%A_%a.out

# CONJUNCTION density, the quantity the BDH paper's ~5% sparsity claim is about
# and the one that actually feeds the decoder.
#
# exp3 measured only the two FACTORS -- x_sparse 21.9%, y_sparse 18.3% -- and
# compared them against GPT's single post-GELU hidden (13.5%), concluding BDH is
# less sparse. But upstream feeds `xy_sparse = x_sparse * y_sparse` to the
# decoder. Two ~20% codes give a conjunction near 4% if close to independent,
# which would match the paper's figure and REVERSE that conclusion. Independence
# is an assumption, so it is measured rather than multiplied.
#
# Same config as exp3 in every other respect, so the numbers are comparable.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8
export WANDB_DIR="$WS/wandb" WANDB_CACHE_DIR="$WS/.wandb-cache"
cd "$WS/code"
echo "=== $(date) job=$SLURM_JOB_ID seed=$SLURM_ARRAY_TASK_ID node=$(hostname) ==="

# bdh_wrapper.py is the file proved equivalent to upstream, so a change to it
# gates the run. Tests on the GPU node take ~30s; on the login node they time
# out, which is why this is here rather than before submission.
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  WANDB_MODE=disabled "$WS/venv/bin/python" -m pytest bdh_cl/tests -q 2>&1 | tail -8
  RC=${PIPESTATUS[0]}
  echo "pytest exit: $RC"
  [ "$RC" != "0" ] && { echo "ABORT: tests failed"; exit 1; }
fi
srun "$WS/venv/bin/python" -u bdh_cl/scripts/sparsity_original.py \
  --model bdh --seed "$SLURM_ARRAY_TASK_ID" \
  --steps 3000 --batch-size 32 --block-size 512 --probe-every 100 \
  --out "$WS/results/exp11_conjunction" \
  --data-cache "$WS/data_cache/shakespeare.txt" \
  --wandb-project bdh-cl --wandb-group exp11_conjunction
echo "=== done $(date) ==="
