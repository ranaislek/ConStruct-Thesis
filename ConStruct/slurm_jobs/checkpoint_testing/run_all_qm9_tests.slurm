#!/bin/bash
#SBATCH --job-name=run_all_qm9_tests
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/checkpoint_testing/run_all_qm9_tests_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G  
#SBATCH --gres=gpu:1
#SBATCH --exclude=vgpu5-0
#SBATCH --time=08:00:00
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

echo "=== Running All QM9 Checkpoint Tests ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Create output directories
mkdir -p /home/rislek/ConStruct-Thesis/ConStruct/outputs/checkpoint_testing
mkdir -p /home/rislek/ConStruct-Thesis/ConStruct/logs/checkpoint_testing

cd ConStruct

# Test 1: QM9 Planar
echo "=== Test 1: QM9 Planar ==="
python main.py \
    +experiment=thesis/edge_deletion/planarity/qm9_thesis_planar \
    general.test_only=checkpoints/qm9_thesis_planar/last.ckpt \
    general.name=test_qm9_planar

if [ $? -eq 0 ]; then
    echo "✅ QM9 Planar test completed successfully!"
else
    echo "❌ QM9 Planar test failed!"
    exit 1
fi

echo "⏳ Waiting 30 seconds before next test..."
sleep 30

# Test 2: QM9 Ring Count ≤ 2
echo "=== Test 2: QM9 Ring Count ≤ 2 ==="
python main.py \
    +experiment=thesis/edge_deletion/ring_count_at_most/qm9_thesis_ring_count_at_most_2 \
    general.test_only=checkpoints/qm9_thesis_ring_count_at_most_2/last.ckpt \
    general.name=test_qm9_ring_count_2

if [ $? -eq 0 ]; then
    echo "✅ QM9 Ring Count ≤ 2 test completed successfully!"
else
    echo "❌ QM9 Ring Count ≤ 2 test failed!"
    exit 1
fi

echo "⏳ Waiting 30 seconds before next test..."
sleep 30

# Test 3: QM9 Ring Length ≤ 4
echo "=== Test 3: QM9 Ring Length ≤ 4 ==="
python main.py \
    +experiment=thesis/edge_deletion/ring_length_at_most/qm9_thesis_ring_length_at_most_4 \
    general.test_only=checkpoints/qm9_thesis_ring_length_at_most_4/last.ckpt \
    general.name=test_qm9_ring_length_4

if [ $? -eq 0 ]; then
    echo "✅ QM9 Ring Length ≤ 4 test completed successfully!"
else
    echo "❌ QM9 Ring Length ≤ 4 test failed!"
    exit 1
fi

echo "⏳ Waiting 30 seconds before next test..."
sleep 30

# Test 4: QM9 No Constraint
echo "=== Test 4: QM9 No Constraint ==="
python main.py \
    +experiment=thesis/no_constraint/qm9_thesis_no_constraint \
    general.test_only=checkpoints/qm9_thesis_no_constraint/last.ckpt \
    general.name=test_qm9_no_constraint

if [ $? -eq 0 ]; then
    echo "✅ QM9 No Constraint test completed successfully!"
else
    echo "❌ QM9 No Constraint test failed!"
    exit 1
fi

echo "🎉 ALL QM9 CHECKPOINT TESTS COMPLETED SUCCESSFULLY!"
echo "Results saved to: outputs/checkpoint_testing/"
echo "Logs saved to: logs/checkpoint_testing/"
echo "Testing completed at: $(date)"
echo ""
echo "📊 Test Summary:"
echo "✅ QM9 Planar - Planarity constraint"
echo "✅ QM9 Ring Count ≤ 2 - Maximum 2 rings"
echo "✅ QM9 Ring Length ≤ 4 - Maximum 4-atom rings"
echo "✅ QM9 No Constraint - Baseline unconstrained" 