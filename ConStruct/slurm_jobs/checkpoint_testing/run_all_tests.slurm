#!/bin/bash
#SBATCH --job-name=run_all_checkpoint_tests
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/checkpoint_testing/run_all_tests_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G  
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu5-0
#SBATCH --time=06:00:00
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

echo "=== Running All Checkpoint Tests ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Create output directories
mkdir -p /home/rislek/ConStruct-Thesis/ConStruct/outputs/checkpoint_testing
mkdir -p /home/rislek/ConStruct-Thesis/ConStruct/logs/checkpoint_testing

cd ConStruct

# Test 1: No Constraint
echo "=== Test 1: No Constraint ==="
python main.py \
    +experiment=checkpoint_testing/qm9_checkpoint_test_no_constraint \
    general.resume=checkpoints/qm9_debug_no_constraint/last.ckpt \
    general.test_only=checkpoints/qm9_debug_no_constraint/last.ckpt

if [ $? -eq 0 ]; then
    echo "✅ No constraint test completed successfully!"
else
    echo "❌ No constraint test failed!"
    exit 1
fi

echo "⏳ Waiting 30 seconds before next test..."
sleep 30

# Test 2: Ring Count ≤ 0
echo "=== Test 2: Ring Count ≤ 0 ==="
python main.py \
    +experiment=checkpoint_testing/qm9_checkpoint_test_ring_count_0 \
    general.resume=checkpoints/qm9_debug_ring_count_at_most_0/last.ckpt \
    general.test_only=checkpoints/qm9_debug_ring_count_at_most_0/last.ckpt

if [ $? -eq 0 ]; then
    echo "✅ Ring count 0 test completed successfully!"
else
    echo "❌ Ring count 0 test failed!"
    exit 1
fi

echo "⏳ Waiting 30 seconds before next test..."
sleep 30

# Test 3: Ring Length ≤ 3
echo "=== Test 3: Ring Length ≤ 3 ==="
python main.py \
    +experiment=checkpoint_testing/qm9_checkpoint_test_ring_length_3 \
    general.resume=checkpoints/qm9_debug_ring_length_at_most_3/last.ckpt \
    general.test_only=checkpoints/qm9_debug_ring_length_at_most_3/last.ckpt

if [ $? -eq 0 ]; then
    echo "✅ Ring length 3 test completed successfully!"
else
    echo "❌ Ring length 3 test failed!"
    exit 1
fi

echo "🎉 ALL CHECKPOINT TESTS COMPLETED SUCCESSFULLY!"
echo "Results saved to: outputs/checkpoint_testing/"
echo "Logs saved to: logs/checkpoint_testing/"
echo "Testing completed at: $(date)"
