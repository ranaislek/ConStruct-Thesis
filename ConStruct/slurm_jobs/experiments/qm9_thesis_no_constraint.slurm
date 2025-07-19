#!/bin/bash
#SBATCH --job-name=qm9_thesis_no_constraint
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/qm9_thesis_no_constraint_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00    # 48 hours for thesis-level experiment
#SBATCH --partition=testing
#SBATCH --exclude=vgpu5-0   # Exclude problematic GPU node

cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# CUDA verification
python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

# Thesis-level experiment: No Constraint Baseline with 1000+ molecule generation
echo "=== Running QM9 Thesis No Constraint Baseline Experiment ==="
echo "Config: 1000 epochs, batch_size=128, 1000 samples, rev_proj=False"
echo "Enhanced model architecture for thesis-level experiments"
python ConStruct/main.py +experiment=qm9_thesis \
  model.rev_proj=False

echo "=== Thesis Baseline Experiment Complete ===" 