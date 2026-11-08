#!/bin/bash
#SBATCH --job-name=bdh-exp22
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-14
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/exp22_%A_%a.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/exp22_%A_%a.out

# Exp22: continual RL on MinAtar, 5 games x 2 cycles x 200k steps.
# Tasks 0-11: {dqn,dqn_redo,pop,pop_tired} x seeds 0-2.
# Tasks 12-14: fresh-agent reference, dqn on each game alone (1 cycle), seed = task-12.
set -uo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
PY="$WS/venv/bin/python"
export OMP_NUM_THREADS=8
cd "$WS/code"
OUT="$WS/results/exp22"; mkdir -p "$OUT"
i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 12 ]; then
  ARMS=(dqn dqn_redo pop pop_tired); arm=${ARMS[$((i / 3))]}; seed=$((i % 3))
  "$PY" bdh_cl/scripts/pop_cl_rl.py --arm $arm --seed $seed --tag ${arm} --out "$OUT/${arm}_s${seed}.json"
else
  seed=$((i - 12))
  for g in breakout space_invaders freeway asterix seaquest; do
    "$PY" bdh_cl/scripts/pop_cl_rl.py --arm dqn --seed $seed --games $g --cycles 1 --tag fresh_$g \
      --out "$OUT/fresh_${g}_s${seed}.json"
  done
fi
echo "=== done $(date) ==="
