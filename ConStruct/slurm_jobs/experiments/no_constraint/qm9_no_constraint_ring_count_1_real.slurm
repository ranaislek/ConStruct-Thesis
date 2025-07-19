#!/bin/bash
#SBATCH --job-name=qm9_no_constraint_ring_count_1
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/qm9_no_constraint_ring_count_1_%j.out
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

# No Constraint Baseline with ring_count=1 analysis
echo "=== Running QM9 No Constraint Ring Count 1 Experiment ==="
echo "Config: 200 epochs, batch_size=128, 200 samples, rev_proj=False, max_rings=1"
echo "Purpose: Train without constraint enforcement but analyze ring_count=1 satisfaction"
python ConStruct/main.py +experiment=qm9_no_constraint_baseline \
  model.max_rings=1 