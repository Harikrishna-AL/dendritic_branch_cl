#!/bin/bash
#SBATCH --job-name=bdh-exp9
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp9_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp9_%A_%a.out

# Milestone 1: persistent synaptic memory, no dendrites, A -> B -> A.
# Array task 0 runs the tests and the smoke config first so a wiring problem
# surfaces before three other GPUs spend an hour on it.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
MODELS=(bdh bdh_mem_slow bdh_mem_fast bdh_mem_nodecay)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=8
export WANDB_DIR="$WS/wandb" WANDB_CACHE_DIR="$WS/.wandb-cache"
mkdir -p "$WS/logs" "$WANDB_DIR" "$WANDB_CACHE_DIR"
cd "$WS/code"
echo "=== $(date) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID model=$MODEL node=$(hostname) ==="

# The tests and the smoke run GATE the experiment. The first version of this
# script printed their results and then ran the sweep regardless -- both of them
# correctly caught an autograd bug that killed every memory arm, and all four
# GPUs proceeded to burn an hour on it anyway. An advisory check is not a check.
echo "########## tests (gating) ##########"
WANDB_MODE=disabled "$WS/venv/bin/python" -m pytest bdh_cl/tests -q 2>&1 | tail -14
TEST_RC=${PIPESTATUS[0]}
echo "pytest exit: $TEST_RC"
if [ "$TEST_RC" != "0" ]; then
  echo "ABORT: tests failed, not running the experiment"; exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  echo "########## smoke (gating) ##########"
  WANDB_MODE=disabled "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
    --config bdh_cl/configs/smoke_persistent.yaml --tag smoke_persistent 2>&1 | tail -16
  SMOKE_RC=${PIPESTATUS[0]}
  echo "smoke exit: $SMOKE_RC"
  if [ "$SMOKE_RC" != "0" ]; then
    echo "ABORT: smoke failed, not running the experiment"; exit 1
  fi
fi

echo "########## exp9: $MODEL ##########"
srun "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
  --config bdh_cl/configs/exp9_persistent.yaml \
  --models "$MODEL" --tag "${TAG:-exp9_persistent}"
echo "=== done $(date) ==="
