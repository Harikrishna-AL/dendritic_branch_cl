#!/bin/bash
#SBATCH --job-name=bdh-exp24
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-23
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp24_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp24_%A_%a.out

# Exp24: which part of the binary top-k layer blocks Q-learning?
# Fresh agent, ONE game, 200k steps. 4 variants x {breakout, space_invaders} x seeds 0-2.
#   pop_k128          binary, 128/512 active            (sparsity)
#   graded_k32        graded output, true gradient      (magnitude + gradient)
#   gradedst_k32      graded output, surrogate gradient (magnitude alone)
#   graded_k128       graded + dense
# References: dqn = exp22 fresh_*, binary k32 = exp23 pop_*.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
PY="$WS/venv/bin/python"
export OMP_NUM_THREADS=8
cd "$WS/code"
i=$SLURM_ARRAY_TASK_ID
[ $i -eq 0 ] && { "$PY" -m pytest -q bdh_cl/tests/test_pop_cl_rl.py || exit 1; }
TAGS=(pop_k128 graded_k32 gradedst_k32 graded_k128)
ARMS=(pop pop_graded pop_gradedst pop_graded); KS=(128 32 32 128)
GAMES=(breakout space_invaders)
v=$((i / 6)); g=${GAMES[$(((i / 3) % 2))]}; seed=$((i % 3))
OUT="$WS/results/exp24"; mkdir -p "$OUT"
"$PY" bdh_cl/scripts/pop_cl_rl.py --arm ${ARMS[$v]} --k ${KS[$v]} --seed $seed --games $g --cycles 1 \
  --group exp24_graded --tag ${TAGS[$v]}_$g --out "$OUT/${TAGS[$v]}_${g}_s${seed}.json"
echo "=== done $(date) ==="
