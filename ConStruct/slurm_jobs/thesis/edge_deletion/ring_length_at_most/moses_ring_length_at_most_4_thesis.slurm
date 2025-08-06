#!/bin/bash
#SBATCH --job-name=moses_ring_length_at_most_4_thesis
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/moses_ring_length_at_most_4_thesis_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu5-0
#SBATCH --constraint=A6000|A5000
#SBATCH --time=36:00:00
#SBATCH --partition=allgroups,testing

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

echo "=== Running MOSES Thesis Experiment: ring_length_at_most_4_thesis ==="
echo "Config: thesis/edge_deletion/ring_length_at_most/moses_thesis_ring_length_at_most_4"
echo "Purpose: Generate MOSES molecules with rings of AT MOST 4 length using edge-deletion mechanism"
echo "Constraint: ring_length_at_most <= 4 (maximum 4-ring length allowed)"
echo "Transition: absorbing_edges (removes edges to satisfy maximum ring length constraints)"
echo "Dataset: MOSES (larger molecules, 25+ nodes, test edge-deletion on complex graphs)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Run the experiment
cd ConStruct
python main.py +experiment=thesis/edge_deletion/ring_length_at_most/moses_thesis_ring_length_at_most_4

echo "Experiment completed at $(date)" 