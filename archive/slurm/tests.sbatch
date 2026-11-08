#!/bin/bash
#SBATCH --job-name=bdh-test
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/data/horse/ws/hapi039h-bdh-d/logs/pytest_%j.out

# Verify the upstream-equivalence proof holds on the cluster's torch build
# before spending GPU time on anything. Run on a compute node, not the login
# node -- login nodes are for editing and submitting.
set -euo pipefail
WS=/data/horse/ws/hapi039h-bdh-d
export OMP_NUM_THREADS=8
mkdir -p "$WS/logs"
cd "$WS/code"

echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$WS/venv/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
"$WS/venv/bin/python" -m pytest bdh_cl/tests -q
