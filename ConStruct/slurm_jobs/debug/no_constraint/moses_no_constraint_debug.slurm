#!/bin/bash
#SBATCH --job-name=moses_no_constraint_debug
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/moses_no_constraint_debug_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G  
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --partition=debug

# Load environment
cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# WandB fixes to prevent hanging
export WANDB_TIMEOUT=300  # 5-minute timeout for WandB operations
export WANDB_START_METHOD=thread  # Use thread-based WandB
export WANDB_CONSOLE=wrap  # Better console output
export WANDB_SILENT=false  # Show WandB progress
export WANDB_NETWORK_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=300

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=1

# Force unbuffered output for better logging
export PYTHONUNBUFFERED=1

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# CUDA verification
python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

echo "=== Running MOSES No-Constraint Experiment ==="
echo "Config: 5 epochs, batch_size=16, 10 samples, standard transition"
echo "Purpose: Generate MOSES molecules without any structural constraints"
echo "Constraint: None (baseline for comparison)"
echo "Transition: absorbing_edges (standard edge-deletion)"
echo "Dataset: MOSES (larger molecules, 25+ nodes, baseline performance)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Run the experiment
cd ConStruct
python main.py +experiment=debug/no_constraint/moses_debug_no_constraint

echo "Experiment completed at $(date)" 