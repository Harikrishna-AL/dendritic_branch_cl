#!/bin/bash
#SBATCH --job-name=bdh-consol2
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/consol2_%j.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/consol2_%j.out

# Stage 2. Stage 1 established that mode="scale" cannot work: AdamW cancels a
# constant gradient rescale, and final accuracy sat on the naive floor (0.197 /
# 0.199) for lam = 1 and 10. This sweeps the Adam-safe modes instead.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8 WANDB_MODE=disabled
cd "$WS/code"
echo "=== $(date) job=$SLURM_JOB_ID node=$(hostname) ==="

echo "########## tests ##########"
"$WS/venv/bin/python" -m pytest bdh_cl/tests -q 2>&1 | tail -15
echo "pytest exit: ${PIPESTATUS[0]}"

echo "########## penalty mode: lam sweep ##########"
for LAM in 1 10 100 1000 10000; do
  echo "----- penalty lam=$LAM -----"
  "$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
    --config bdh_cl/configs/calib_consol.yaml \
    --consol-mode penalty --consol-lam "$LAM" \
    --tag "calib_pen_lam$LAM" --quiet 2>&1 | tail -5
done

echo "########## scale mode at lam=1000, the clean negative control ##########"
# Stage 1's lam=1 and lam=10 scale runs were valid (0.197 / 0.199, the naive
# floor). Its lam=100 and lam=1000 runs were contaminated -- new code was
# uploaded mid-sweep and each lam is a separate process -- so they are
# discarded rather than mislabelled, and the strongest scale case is rerun here.
# If a 1000x gradient reduction still cannot lift the floor, scale-invariance
# under Adam is confirmed by experiment and not only by the unit test.
"$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
  --config bdh_cl/configs/calib_consol.yaml \
  --consol-mode scale --consol-lam 1000 \
  --tag calib_scale_lam1000 --quiet 2>&1 | tail -5

echo "########## mask mode (hard freeze), for contrast ##########"
"$WS/venv/bin/python" -u bdh_cl/scripts/run_cl.py \
  --config bdh_cl/configs/calib_consol.yaml \
  --consol-mode mask --tag calib_mask --quiet 2>&1 | tail -5

echo "=== done $(date) ==="
