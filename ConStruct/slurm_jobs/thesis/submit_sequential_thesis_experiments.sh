#!/bin/bash

# Script to submit thesis experiments to queue immediately (no waiting)
echo "=== Submitting Sequential Thesis Experiments to Queue ==="
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
submit_job "no_constraint/qm9_no_constraint_marginal_thesis.slurm" "qm9_no_constraint_marginal_thesis" "QM9 No Constraint (Marginal) - Thesis"

# 2. No constraint normal
submit_job "no_constraint/qm9_no_constraint_thesis.slurm" "qm9_no_constraint_thesis" "QM9 No Constraint (Normal) - Thesis"

# ============================================================================
# PHASE 2: Planarity Experiments (Queue Submission)
# ============================================================================
echo "=== PHASE 2: Planarity Experiments ==="

# 3. QM9 planar
submit_job "edge_deletion/planarity/qm9_thesis_planar.slurm" "qm9_thesis_planar" "QM9 Planar Constraint - Thesis"

# 4. Planar planar
submit_job "edge_deletion/planarity/planar_thesis_planar.slurm" "planar_thesis_planar" "Planar Planar Constraint - Thesis"

# ============================================================================
# PHASE 3: Ring Count and Ring Length Experiments (Alternating Queue Submission)
# ============================================================================
echo "=== PHASE 3: Ring Count and Ring Length Experiments (Alternating) ==="

# 5. Ring count at most 0
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_0_thesis.slurm" "qm9_ring_count_at_most_0_thesis" "QM9 Ring Count ≤ 0 - Thesis"

# 6. Ring length at most 3
submit_job "edge_deletion/ring_length_at_most/qm9_ring_length_at_most_3_thesis.slurm" "qm9_ring_length_at_most_3_thesis" "QM9 Ring Length ≤ 3 - Thesis"

# 7. Ring count at most 1
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_1_thesis.slurm" "qm9_ring_count_at_most_1_thesis" "QM9 Ring Count ≤ 1 - Thesis"

# 8. Ring length at most 4
submit_job "edge_deletion/ring_length_at_most/qm9_ring_length_at_most_4_thesis.slurm" "qm9_ring_length_at_most_4_thesis" "QM9 Ring Length ≤ 4 - Thesis"

# 9. Ring count at most 2
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_2_thesis.slurm" "qm9_ring_count_at_most_2_thesis" "QM9 Ring Count ≤ 2 - Thesis"

# 10. Ring length at most 5
submit_job "edge_deletion/ring_length_at_most/qm9_ring_length_at_most_5_thesis.slurm" "qm9_ring_length_at_most_5_thesis" "QM9 Ring Length ≤ 5 - Thesis"

# 11. Ring count at most 3
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_3_thesis.slurm" "qm9_ring_count_at_most_3_thesis" "QM9 Ring Count ≤ 3 - Thesis"

# 12. Ring length at most 6
submit_job "edge_deletion/ring_length_at_most/qm9_ring_length_at_most_6_thesis.slurm" "qm9_ring_length_at_most_6_thesis" "QM9 Ring Length ≤ 6 - Thesis"

# 13. Ring count at most 4
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_4_thesis.slurm" "qm9_ring_count_at_most_4_thesis" "QM9 Ring Count ≤ 4 - Thesis"

# 14. Ring count at most 5
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_5_thesis.slurm" "qm9_ring_count_at_most_5_thesis" "QM9 Ring Count ≤ 5 - Thesis"

echo "=== All Thesis Experiments Submitted to Queue ==="
echo "Time: $(date)"
echo "Total experiments submitted: 14"
echo "Check job status with: squeue -u rislek"
echo "Jobs will run in the order they were submitted (alternating ring count/ring length)" 