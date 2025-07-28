#!/bin/bash
#SBATCH --job-name=moses_ring_length_at_least_5_debug
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/moses_ring_length_at_least_5_debug_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G  
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --partition=debug

# Load environment
cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct-env

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

echo "=== Running MOSES Edge-Insertion Ring Length At Least 5 Experiment ==="
echo "Config: 5 epochs, batch_size=16, 10 samples, edge-insertion transition"
echo "Purpose: Generate MOSES molecules with rings of AT LEAST 5 length using edge-insertion mechanism"
echo "Constraint: ring_length_at_least >= 5 (minimum 5-ring length required)"
echo "Transition: edge_insertion (adds edges to satisfy minimum ring length constraints)"
echo "Dataset: MOSES (larger molecules, 25+ nodes, better for edge-insertion)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Run the experiment
cd ConStruct
python main.py +experiment=debug/edge_insertion/ring_length_at_least/moses_debug_ring_length_at_least_5

echo "Experiment completed at $(date)" 