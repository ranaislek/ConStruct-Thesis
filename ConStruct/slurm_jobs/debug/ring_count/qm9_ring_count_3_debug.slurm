#!/bin/bash
#SBATCH --job-name=debug_qm9_ring_count_3
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/debug_qm9_ring_count_3_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=debug
#SBATCH --time=00:30:00    # 30 minutes max

cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# CUDA verification
python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

# Quick test with debug config (1 epoch, small batch, few samples)
echo "=== Testing Ring Count Constraint: 3 rings (current choice) ==="
echo "Using qm9_debug config: 1 epoch, batch_size=4, 5 samples, max_rings=3"
python ConStruct/main.py +experiment=qm9_debug \
  model.rev_proj=ring_count \
  model.max_rings=3

echo "=== Test Complete ===" 