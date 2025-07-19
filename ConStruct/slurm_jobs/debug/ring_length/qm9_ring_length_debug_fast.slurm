#!/bin/bash
#SBATCH --job-name=debug_qm9_ring_count_fast
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/debug_qm9_ring_count_fast_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=debug
#SBATCH --time=00:05:00    # 5 minutes max

cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# CUDA verification
python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

# Super-fast test with debug_fast config (1 epoch, tiny batch, 3 samples, 5 diffusion steps)
echo "=== Testing Ring Count Constraint (Super Fast Debug) ==="
echo "Using qm9_debug_fast config: 1 epoch, batch_size=2, 3 samples, 5 diffusion steps, subset=10"
python ConStruct/main.py +experiment=qm9_debug_fast \
  model.rev_proj=ring_length \
  model.max_ring_length=6

echo "=== Test Complete ===" 