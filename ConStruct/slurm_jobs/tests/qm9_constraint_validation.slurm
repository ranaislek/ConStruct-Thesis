#!/bin/bash
#SBATCH --job-name=qm9_constraint_validation
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/qm9_constraint_validation_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=debug
#SBATCH --time=00:15:00    # 15 minutes max

cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# CUDA verification
python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

echo "=== QM9 Constraint Validation Test ==="
echo "Testing both ring_count and ring_length constraints"

# Test 1: Ring Count Constraint
echo "--- Testing Ring Count Constraint (max 0 rings) ---"
python ConStruct/main.py +experiment=qm9_debug \
  model.rev_proj=ring_count \
  model.max_rings=0

echo "--- Ring Count Test Complete ---"

# Test 2: Ring Length Constraint  
echo "--- Testing Ring Length Constraint (max 6 atoms per ring) ---"
python ConStruct/main.py +experiment=qm9_debug \
  model.rev_proj=ring_length \
  model.max_ring_length=6

echo "--- Ring Length Test Complete ---"
echo "=== All Constraint Tests Complete ===" 