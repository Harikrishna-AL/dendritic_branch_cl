#!/bin/bash
#SBATCH --job-name=bdh-exp23
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-17
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp23_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp23_%A_%a.out

# Exp23: can homeostatic excitability make the binary population layer learn?
# Fresh agent, ONE game, 200k steps (no continual learning).
# {pop, pop_homeo, pop_homeo_up} x {breakout, space_invaders} x seeds 0-2.
# dqn reference = exp22 fresh_* runs (identical settings).
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
PY="$WS/venv/bin/python"
export OMP_NUM_THREADS=8
cd "$WS/code"
i=$SLURM_ARRAY_TASK_ID
[ $i -eq 0 ] && { "$PY" -m pytest -q bdh_cl/tests/test_pop_cl_rl.py || exit 1; }
ARMS=(pop pop_homeo pop_homeo_up); GAMES=(breakout space_invaders)
arm=${ARMS[$((i / 6))]}; g=${GAMES[$(((i / 3) % 2))]}; seed=$((i % 3))
OUT="$WS/results/exp23"; mkdir -p "$OUT"
"$PY" bdh_cl/scripts/pop_cl_rl.py --arm $arm --seed $seed --games $g --cycles 1 \
  --group exp23_homeo --tag ${arm}_$g --out "$OUT/${arm}_${g}_s${seed}.json"
echo "=== done $(date) ==="
