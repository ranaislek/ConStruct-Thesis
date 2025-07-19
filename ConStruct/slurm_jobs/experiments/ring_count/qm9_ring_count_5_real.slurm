#!/bin/bash
#SBATCH --job-name=qm9_ring_count_5_real
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/qm9_ring_count_5_real_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=debug

cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# CUDA verification
python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

# Real experiment: Ring Count Constraint (max 5 rings - natural dataset limit)
echo "=== Running QM9 Ring Count 5 Real Experiment ==="
echo "Config: 200 epochs, batch_size=128, 100 samples, max_rings=5"
echo "Purpose: Natural dataset limit - only 9 molecules violate this constraint"
python ConStruct/main.py +experiment=qm9 \
  model.rev_proj=ring_count \
  model.max_rings=5 