#!/bin/bash
#SBATCH --job-name=qm9_no_constraint_baseline_debug
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/qm9_no_constraint_baseline_debug_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --partition=debug

cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# CUDA verification
python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

# Debug experiment: No Constraint Enforcement (baseline)
echo "=== Running QM9 No Constraint Baseline Debug Experiment ==="
echo "Config: 1 epoch, batch_size=4, 10 samples, rev_proj=False"
echo "Purpose: Quick test of no constraint enforcement baseline"
python ConStruct/main.py +experiment=qm9_no_constraint_baseline_debug 