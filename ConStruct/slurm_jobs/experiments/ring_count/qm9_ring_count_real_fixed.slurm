#!/bin/bash
#SBATCH --job-name=qm9_ring_count_real_fixed
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/qm9_ring_count_real_fixed_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000|A5000
#SBATCH --time=24:00:00
#SBATCH --partition=allgroups,testing

# Set working directory
cd /home/rislek/ConStruct-Thesis

# Force conda environment activation
export CONDA_BASE=/conf/shared-software/anaconda
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate construct

# Verify environment
echo "=== Environment Check ==="
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# CUDA verification with detailed diagnostics
echo "=== CUDA Diagnostics ==="
python -c '
import torch
import os
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability(0))
else:
    print("CUDA not available - checking environment...")
    import subprocess
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print("nvidia-smi output:", result.stdout[:500])
    except Exception as e:
        print("nvidia-smi error:", e)
'

# Force GPU usage if available
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "CUDA is available - proceeding with GPU training"
    export CUDA_VISIBLE_DEVICES=0
else
    echo "WARNING: CUDA not available - job will run on CPU"
    echo "This may be due to driver/library version mismatch"
    echo "Job will continue but will be much slower"
fi

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# Real experiment: Ring Count Constraint (max 3 rings)
echo "=== Running QM9 Ring Count Real Experiment ==="
echo "Config: 200 epochs, batch_size=128, 100 samples, max_rings=3"
python ConStruct/main.py +experiment=qm9 \
  model.rev_proj=ring_count \
  model.max_rings=3 