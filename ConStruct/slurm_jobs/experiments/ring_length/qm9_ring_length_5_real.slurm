#!/bin/bash
#SBATCH --job-name=qm9_ring_length_5_real
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/qm9_ring_length_5_real_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000|A5000
#SBATCH --time=24:00:00
#SBATCH --partition=allgroups,testing

cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# CUDA verification
python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

# Real experiment: Ring Length Constraint (max 5 atoms per ring)
echo "=== Running QM9 Ring Length 5 Real Experiment ==="
echo "Config: 200 epochs, batch_size=128, 100 samples, max_ring_length=5"
echo "Purpose: Up to 5-membered rings constraint - covers most common rings"
python ConStruct/main.py +experiment=qm9 \
  model.rev_proj=ring_length \
  model.max_ring_length=5 