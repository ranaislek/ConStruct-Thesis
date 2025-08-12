#!/bin/bash
#SBATCH --job-name=test_qm9_planar
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/checkpoint_testing/test_qm9_planar_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G  
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu5-0
#SBATCH --time=02:00:00
#SBATCH --partition=debug

# Load environment
cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct-env

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# WandB fixes to prevent hanging
export WANDB_TIMEOUT=300
export WANDB_START_METHOD=thread
export WANDB_CONSOLE=wrap
export WANDB_SILENT=false
export WANDB_NETWORK_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=300

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# CUDA verification
python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

echo "=== Testing QM9 Planar Checkpoint ==="
echo "Config: thesis/edge_deletion/planarity/qm9_thesis_planar"
echo "Checkpoint: checkpoints/qm9_thesis_planar/last.ckpt"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Create output directories
mkdir -p /home/rislek/ConStruct-Thesis/ConStruct/outputs/checkpoint_testing/qm9_planar
mkdir -p /home/rislek/ConStruct-Thesis/ConStruct/logs/checkpoint_testing

# Run the experiment with TESTING ONLY (no training)
cd ConStruct
python main.py \
    +experiment=thesis/edge_deletion/planarity/qm9_thesis_planar \
    general.test_only=checkpoints/qm9_thesis_planar/last.ckpt \
    general.name=test_qm9_planar

echo "Experiment completed at $(date)"
echo "Results saved to: outputs/checkpoint_testing/qm9_planar/" 