#!/bin/bash
#SBATCH --job-name=bdh-exp25
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --array=0-11
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp25_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp25_%A_%a.out

# Exp25: continual RL (5 MinAtar games x 2 cycles x 200k steps) with the graded
# top-k population layer, k=128/512, 4 branches.
# {pop_graded, pop_graded_tired, pop_graded_reset, pop_graded_tired_reset} x seeds 0-2.
# References (exp22, same protocol): dqn, dqn_redo, fresh_*.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
PY="$WS/venv/bin/python"
export OMP_NUM_THREADS=8
cd "$WS/code"
i=$SLURM_ARRAY_TASK_ID
[ $i -eq 0 ] && { "$PY" -m pytest -q bdh_cl/tests/test_pop_cl_rl.py || exit 1; }
ARMS=(pop_graded pop_graded_tired pop_graded_reset pop_graded_tired_reset)
arm=${ARMS[$((i / 3))]}; seed=$((i % 3))
OUT="$WS/results/exp25"; mkdir -p "$OUT"
"$PY" bdh_cl/scripts/pop_cl_rl.py --arm $arm --k 128 --seed $seed \
  --group exp25_graded_cl --tag ${arm}_k128 --out "$OUT/${arm}_k128_s${seed}.json"
echo "=== done $(date) ==="
