#!/bin/bash
#SBATCH --job-name=bdh-proto18
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/proto18_%j.out
#SBATCH --error=/data/horse/ws/hapi039h-bdh-d/logs/proto18_%j.out

# Nearest-class-mean readout on the exp18 (SET rewire) and exp19 (coactivity
# rewire) checkpoints. Eval only, no training. Question: do sparse/rewired
# dendritic branches keep features the learned head throws away?
set -uo pipefail
# Rerun of exp18 after proto_eval.load learned to zero-fill the coactivity
# buffer that exp18-era sparse checkpoints lack (job 4384314 crashed there).
WS=/data/horse/ws/hapi039h-bdh-d
PY="$WS/venv/bin/python"
export OMP_NUM_THREADS=8 WANDB_MODE=disabled
cd "$WS/code"
echo "=== $(date) job=$SLURM_JOB_ID node=$(hostname) ==="
WANDB_MODE=disabled "$PY" -m pytest bdh_cl/tests/test_proto_eval.py -q || exit 1


run() {  # exp config models seed
  "$PY" -u bdh_cl/scripts/proto_eval.py --config "bdh_cl/configs/$1.yaml" \
    --ckpt-dir "bdh_cl/results/$1_seed$3/ckpt" --models "$2" \
    --strategies naive,replay --seeds "$3" --out "$WS/results/proto_$1_seed$3.json"
}
M18=bdh_narrow,bdh_den_max,bdh_den_point_sparse_fixed,bdh_den_point_rewire,bdh_den_sparse_fixed,bdh_den_rewire
M19=bdh_narrow,bdh_den_max,bdh_den_point_sparse_fixed,bdh_den_point_coactivity_rewire,bdh_den_sparse_fixed,bdh_den_coactivity_rewire
for s in 0 1 2; do
  echo "########## exp18 seed$s ##########"; run exp18_sparse_rewire "$M18" "$s"

done
echo "=== done $(date) ==="
