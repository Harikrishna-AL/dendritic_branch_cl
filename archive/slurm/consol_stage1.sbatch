#!/bin/bash
#SBATCH --job-name=bdh-consol1
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/consol1_%j.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/consol1_%j.out

# Stage 1 for branch-granular consolidation, in the order the ground rules ask:
#   1. full test suite  -- runner.py and run_cl.py were both modified, so this
#      also guards against breaking every existing experiment
#   2. smoke config     -- the pipeline end to end on tiny before any real run
#   3. lam calibration  -- lam decides whether the idea works at all, so it is
#      measured on one seed before the 24-run sweep is queued
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8 WANDB_MODE=disabled
cd "$WS/code"
echo "=== $(date) job=$SLURM_JOB_ID node=$(hostname) ==="

echo "########## 1. full test suite ##########"
"$WS/venv/bin/python" -m pytest bdh_cl/tests -q 2>&1 | tail -25
echo "pytest exit: ${PIPESTATUS[0]}"

echo "########## 2. smoke: consolidation end to end ##########"
"$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
  --config bdh_cl/configs/smoke_consol.yaml --tag smoke_consol 2>&1 | tail -30

echo "########## 3. lam calibration on split MNIST ##########"
for LAM in 1 10 100 1000; do
  echo "----- lam=$LAM -----"
  "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
    --config bdh_cl/configs/calib_consol.yaml \
    --consol-lam "$LAM" --tag "calib_consol_lam$LAM" --quiet 2>&1 | tail -6
done

echo "=== done $(date) ==="
