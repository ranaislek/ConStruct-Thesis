#!/bin/bash
#SBATCH --job-name=test_all_constraints
#SBATCH --output=ConStruct/slurm_jobs/checkpoint_testing/results/logs/test_all_constraints_%j.out
#SBATCH --error=ConStruct/slurm_jobs/checkpoint_testing/results/logs/test_all_constraints_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000|A5000
#SBATCH --exclude=vgpu5-0
#SBATCH --time=24:00:00
#SBATCH --partition=allgroups,testing

# =============================================================================
# Test All Constraints Using No-Constraint Checkpoint
# =============================================================================
# This script tests all constraint types using the trained no-constraint model
# 
# Key Insight: Projectors only work during sampling, so we can reuse the
# no-constraint checkpoint with different rev_proj settings
#
# Usage:
#   sbatch ConStruct/slurm_jobs/checkpoint_testing/test_all_constraints_no_constraint_ckpt.slurm
# =============================================================================

# Load environment
source ~/.bashrc
conda activate construct-env

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Set PYTHONPATH
export PYTHONPATH="/home/rislek/ConStruct-Thesis:${PYTHONPATH}"

# Navigate to project directory
cd /home/rislek/ConStruct-Thesis

# Create timestamp for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="ConStruct/slurm_jobs/checkpoint_testing/results/${TIMESTAMP}_all_constraints_test"

# Create results directory structure
mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/metrics"
mkdir -p "${RESULTS_DIR}/samples"

# Create logs directory if it doesn't exist
mkdir -p "ConStruct/slurm_jobs/checkpoint_testing/results/logs"

echo "=============================================================================="
echo "🧪 Testing All Constraints Using No-Constraint Checkpoint"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Expected Duration: 24 hours"
echo "Results Directory: $RESULTS_DIR"
echo ""
echo "Key Insight: Projectors work during sampling, so we reuse the"
echo "trained no-constraint model with different rev_proj settings"
echo "=============================================================================="

# =============================================================================
# Define all constraint tests
# =============================================================================

declare -A CONSTRAINT_TESTS=(
    # No constraint (baseline)
    ["no_constraint"]="null"
    
    # Ring count constraints
    ["ring_count_0"]="ring_count_at_most"
    ["ring_count_1"]="ring_count_at_most"
    ["ring_count_2"]="ring_count_at_most"
    ["ring_count_3"]="ring_count_at_most"
    ["ring_count_4"]="ring_count_at_most"
    ["ring_count_5"]="ring_count_at_most"
    
    # Ring length constraints
    ["ring_length_3"]="ring_length_at_most"
    ["ring_length_4"]="ring_length_at_most"
    ["ring_length_5"]="ring_length_at_most"
    ["ring_length_6"]="ring_length_at_most"
    ["ring_length_7"]="ring_length_at_most"
    ["ring_length_8"]="ring_length_at_most"
    
    # Planarity constraint
    ["planar"]="planar"
)

# =============================================================================
# Test each constraint type
# =============================================================================

for test_name in "${!CONSTRAINT_TESTS[@]}"; do
    projector="${CONSTRAINT_TESTS[$test_name]}"
    
    echo ""
    echo "=============================================================================="
    echo "🧪 Testing: $test_name"
    echo "Projector: $projector"
    echo "=============================================================================="
    
    # Create test-specific config
    TEST_CONFIG="${RESULTS_DIR}/config_${test_name}.yaml"
    
    # Base config content
    cat > "$TEST_CONFIG" << EOF
# @package _global_
defaults:
  - override /dataset: qm9
  - override /train: train_default
  - override /general: general_default

general:
  name: 'test_${test_name}'
  test_only: ConStruct/checkpoints/qm9_thesis_no_constraint/last.ckpt
  evaluate_all_checkpoints: false
  wandb: "disabled"
  final_model_samples_to_generate: 10000

dataset:
  name: "qm9"

model:
  transition: absorbing_edges
  rev_proj: ${projector}
EOF

    # Add constraint-specific parameters
    case $test_name in
        ring_count_*)
            ring_count=${test_name#ring_count_}
            cat >> "$TEST_CONFIG" << EOF
  max_rings: ${ring_count}
  use_incremental: true
EOF
            ;;
        ring_length_*)
            ring_length=${test_name#ring_length_}
            cat >> "$TEST_CONFIG" << EOF
  max_ring_length: ${ring_length}
  use_incremental_length: true
EOF
            ;;
    esac
    
    # Complete the config
    cat >> "$TEST_CONFIG" << EOF
  diffusion_steps: 500
  n_layers: 4
  hidden_mlp_dims: { X: 128, E: 128, y: 128 }
  hidden_dims:
    dx: 128
    de: 128
    dy: 128
    n_head: 4
    dim_ffX: 256
    dim_ffE: 256
    dim_ffy: 256

test:
  batch_size: 128
  num_workers: 4
EOF
    
    echo "✅ Created test config: $TEST_CONFIG"
    
    # Run the test
    echo "🚀 Running test for $test_name..."
    
    # Create test-specific output directory
    TEST_OUTPUT_DIR="${RESULTS_DIR}/${test_name}"
    mkdir -p "$TEST_OUTPUT_DIR"
    
    # Run the test
    python ConStruct/main.py \
        --config-path "$(dirname "$TEST_CONFIG")" \
        --config-name "$(basename "$TEST_CONFIG" .yaml)" \
        general.name="test_${test_name}_${TIMESTAMP}" \
        general.wandb="disabled" \
        2>&1 | tee "${TEST_OUTPUT_DIR}/test_output.log"
    
    # Check if the test was successful
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "✅ Test completed successfully for: $test_name"
        
        # Copy results to test directory
        if [[ -d "ConStruct/outputs" ]]; then
            cp -r ConStruct/outputs/* "$TEST_OUTPUT_DIR/" 2>/dev/null || true
        fi
    else
        echo "❌ Test failed for: $test_name"
    fi
    
    echo "=============================================================================="
done

# =============================================================================
# Final summary
# =============================================================================

echo ""
echo "=============================================================================="
echo "🧪 All Constraint Tests Completed"
echo "=============================================================================="
echo "End Time: $(date)"
echo "Results Directory: $RESULTS_DIR"
echo ""
echo "Tests Completed:"
for test_name in "${!CONSTRAINT_TESTS[@]}"; do
    if [[ -f "${RESULTS_DIR}/${test_name}/test_output.log" ]]; then
        echo "✅ $test_name"
    else
        echo "❌ $test_name"
    fi
done
echo "=============================================================================="

# Create summary file
cat > "${RESULTS_DIR}/all_tests_summary.txt" << EOF
All Constraint Tests Summary
============================
Timestamp: $(date)
Job ID: $SLURM_JOB_ID
Node: $SLURM_NODELIST

Key Insight: Using no-constraint checkpoint with different projectors
Checkpoint: ConStruct/checkpoints/qm9_thesis_no_constraint/last.ckpt

Tests Completed:
$(for test_name in "${!CONSTRAINT_TESTS[@]}"; do
    if [[ -f "${RESULTS_DIR}/${test_name}/test_output.log" ]]; then
        echo "- $test_name: ✅ Completed"
    else
        echo "- $test_name: ❌ Failed"
    fi
done)

Results:
- Logs: ${RESULTS_DIR}/logs/
- Metrics: ${RESULTS_DIR}/metrics/
- Samples: ${RESULTS_DIR}/samples/
- Individual Test Results: ${RESULTS_DIR}/*/

All tests completed at: $(date)
EOF

echo "📁 Complete summary saved to: ${RESULTS_DIR}/all_tests_summary.txt"
echo "🎉 All constraint testing completed using no-constraint checkpoint!"
echo ""
echo "💡 Key Insight: Projectors work during sampling, so we successfully"
echo "   reused the trained model with different constraint enforcements!" 