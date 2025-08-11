#!/bin/bash
#SBATCH --job-name=qm9_ring_length_at_most_3_simple_debug
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/debug/qm9_ring_length_at_most_3_simple_debug_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G  
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu5-0
#SBATCH --time=16:00:00
#SBATCH --partition=debug

# Load environment
cd /home/rislek/ConStruct-Thesis

# Only activate conda if not already in the right environment
if [ "$CONDA_DEFAULT_ENV" != "construct-env" ]; then
    source /conf/shared-software/anaconda/etc/profile.d/conda.sh
    conda activate construct-env
fi

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# WandB fixes to prevent hanging
export WANDB_TIMEOUT=300
export WANDB_START_METHOD=thread
export WANDB_CONSOLE=wrap
export WANDB_SILENT=false
export WANDB_NETWORK_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=300

# Force unbuffered output for better logging
export PYTHONUNBUFFERED=1

echo "=== Running Experiment: ring_length_at_most_3_simple_debug ==="
echo "Config: debug/edge_deletion/ring_length_at_most/qm9_debug_ring_length_at_most_3_simple"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Run the experiment
cd ConStruct
python main.py +experiment=debug/edge_deletion/ring_length_at_most/qm9_debug_ring_length_at_most_3_simple

echo "Experiment completed at $(date)" 