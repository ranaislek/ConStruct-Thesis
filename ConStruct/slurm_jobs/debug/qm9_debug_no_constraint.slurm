#!/bin/bash
#SBATCH --job-name=qm9_debug_no_constraint
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/qm9_debug_no_constraint_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00    # 3 hours for debug experiment
#SBATCH --partition=debug
#SBATCH --exclude=vgpu5-0   # Exclude problematic GPU node

cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# CUDA verification
python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

# Debug experiment: No Constraint Baseline with fast settings
echo "=== Running QM9 Debug No Constraint Baseline Experiment ==="
echo "Config: 10 epochs, batch_size=8, 10 samples, rev_proj=False"
echo "Purpose: Quick validation of baseline generation (no constraints)"
python ConStruct/main.py +experiment=qm9_debug \
  model.rev_proj=False

echo "=== Debug Baseline Experiment Complete ===" 