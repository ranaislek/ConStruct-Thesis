#!/bin/bash

# Script to submit experiments to queue immediately (no waiting)
echo "=== Submitting Sequential Experiments to Queue ==="
echo "Time: $(date)"

# Function to submit jobs with error handling and wait for completion
submit_and_wait() {
    local script_path="$1"
    local job_name="$2"
    local description="$3"
    
    echo "--- Submitting $description ---"
    echo "Submitting $job_name..."
    job_id=$(sbatch "$script_path" | awk '{print $4}')
    if [ $? -eq 0 ]; then
        echo "✅ $job_name submitted with ID: $job_id"
        echo "Waiting for $job_name to complete..."
        
        # Wait for job to complete
        while squeue -j $job_id 2>/dev/null | grep -q $job_id; do
            echo "⏳ $job_name (ID: $job_id) still running..."
            sleep 60  # Check every minute
        done
        
        echo "✅ $job_name completed!"
    else
        echo "❌ Failed to submit $job_name"
        return 1
    fi
}

# Function to submit jobs without waiting (for parallel execution)
submit_job() {
    local script_path="$1"
    local job_name="$2"
    local description="$3"
    
    echo "--- Submitting $description ---"
    echo "Submitting $job_name..."
    job_id=$(sbatch "$script_path" | awk '{print $4}')
    if [ $? -eq 0 ]; then
        echo "✅ $job_name submitted with ID: $job_id"
    else
        echo "❌ Failed to submit $job_name"
        return 1
    fi
}

# ============================================================================
# PHASE 1: No Constraint Experiments (Queue Submission)
# ============================================================================
echo "=== PHASE 1: No Constraint Experiments ==="

# 1. No constraint marginal
submit_job "no_constraint/qm9_no_constraint_marginal_debug.slurm" "qm9_no_constraint_marginal_debug" "QM9 No Constraint (Marginal)"

# 2. No constraint normal
submit_job "no_constraint/qm9_no_constraint_debug.slurm" "qm9_no_constraint_debug" "QM9 No Constraint (Normal)"

# ============================================================================
# PHASE 2: Planarity Experiments (Queue Submission)
# ============================================================================
echo "=== PHASE 2: Planarity Experiments ==="

# 3. QM9 planar
submit_job "edge_deletion/planarity/qm9_debug_planar.slurm" "qm9_debug_planar" "QM9 Planar Constraint"

# 4. Planar planar (if exists, otherwise skip)
if [ -f "edge_deletion/planarity/planar_debug_planar.slurm" ]; then
    submit_job "edge_deletion/planarity/planar_debug_planar.slurm" "planar_debug_planar" "Planar Planar Constraint"
else
    echo "⚠️  Planar planar experiment not found, skipping..."
fi

# ============================================================================
# PHASE 3: Ring Count and Ring Length Experiments (Alternating Queue Submission)
# ============================================================================
echo "=== PHASE 3: Ring Count and Ring Length Experiments (Alternating) ==="

# 5. Ring count at most 0
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_0_debug.slurm" "qm9_ring_count_at_most_0_debug" "QM9 Ring Count ≤ 0"

# 6. Ring length at most 3
submit_job "edge_deletion/ring_length_at_most/qm9_ring_length_at_most_3_debug.slurm" "qm9_ring_length_at_most_3_debug" "QM9 Ring Length ≤ 3"

# 7. Ring count at most 1
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_1_debug.slurm" "qm9_ring_count_at_most_1_debug" "QM9 Ring Count ≤ 1"

# 8. Ring length at most 4
submit_job "edge_deletion/ring_length_at_most/qm9_ring_length_at_most_4_debug.slurm" "qm9_ring_length_at_most_4_debug" "QM9 Ring Length ≤ 4"

# 9. Ring count at most 2
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_2_debug.slurm" "qm9_ring_count_at_most_2_debug" "QM9 Ring Count ≤ 2"

# 10. Ring length at most 5
submit_job "edge_deletion/ring_length_at_most/qm9_ring_length_at_most_5_debug.slurm" "qm9_ring_length_at_most_5_debug" "QM9 Ring Length ≤ 5"

# 11. Ring count at most 3
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_3_debug.slurm" "qm9_ring_count_at_most_3_debug" "QM9 Ring Count ≤ 3"

# 12. Ring length at most 6
submit_job "edge_deletion/ring_length_at_most/qm9_ring_length_at_most_6_debug.slurm" "qm9_ring_length_at_most_6_debug" "QM9 Ring Length ≤ 6"

# 13. Ring count at most 4
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_4_debug.slurm" "qm9_ring_count_at_most_4_debug" "QM9 Ring Count ≤ 4"

# 14. Ring count at most 5
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_5_debug.slurm" "qm9_ring_count_at_most_5_debug" "QM9 Ring Count ≤ 5"

echo "=== All Debug Experiments Submitted to Queue ==="
echo "Time: $(date)"
echo "Total experiments submitted: 14"
echo "Check job status with: squeue -u rislek"
echo "Jobs will run in the order they were submitted (alternating ring count/ring length)" 