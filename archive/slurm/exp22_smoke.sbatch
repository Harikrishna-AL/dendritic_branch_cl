#!/bin/bash
#SBATCH --job-name=bdh-exp22-smoke
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp22smoke_%j.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp22smoke_%j.out

# Exp22 smoke: every arm, 2 games x 10k steps; also measures env steps/s.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
PY="$WS/venv/bin/python"
export OMP_NUM_THREADS=8
cd "$WS/code"
"$PY" -m pytest -q bdh_cl/tests/test_pop_cl_rl.py || exit 1
for arm in dqn dqn_redo pop pop_tired pop_reset pop_tired_reset; do
  echo "=== $arm ==="
  "$PY" bdh_cl/scripts/pop_cl_rl.py --arm $arm --seed 0 --games breakout,freeway --cycles 1 \
     --steps 10000 --eval-episodes 2 --no-wandb | grep -E "PHASE|Error|Traceback" || echo "FAILED $arm"
done
echo "=== done $(date) ==="
