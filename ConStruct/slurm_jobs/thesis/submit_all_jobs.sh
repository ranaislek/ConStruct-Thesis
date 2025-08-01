#!/bin/bash

# Script to submit all thesis jobs
echo "=== Submitting All Thesis Jobs ==="
echo "Time: $(date)"

# Function to submit jobs with error handling
submit_job() {
    local script_path="$1"
    local job_name="$2"
    
    echo "Submitting $job_name..."
    job_id=$(sbatch "$script_path" | awk '{print $4}')
    if [ $? -eq 0 ]; then
        echo "✅ $job_name submitted with ID: $job_id"
    else
        echo "❌ Failed to submit $job_name"
    fi
}

# No constraint experiments
echo "--- Submitting No Constraint Experiments ---"
submit_job "no_constraint/qm9_no_constraint_thesis.slurm" "qm9_no_constraint_thesis"

# Planarity experiments
echo "--- Submitting Planarity Experiments ---"
submit_job "edge_deletion/planarity/qm9_thesis_planar.slurm" "qm9_thesis_planar"

# Ring count at most experiments
echo "--- Submitting Ring Count At Most Experiments ---"
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_0_thesis.slurm" "qm9_ring_count_at_most_0_thesis"
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_1_thesis.slurm" "qm9_ring_count_at_most_1_thesis"
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_2_thesis.slurm" "qm9_ring_count_at_most_2_thesis"
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_3_thesis.slurm" "qm9_ring_count_at_most_3_thesis"
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_4_thesis.slurm" "qm9_ring_count_at_most_4_thesis"
submit_job "edge_deletion/ring_count_at_most/qm9_ring_count_at_most_5_thesis.slurm" "qm9_ring_count_at_most_5_thesis"

# Ring length at most experiments
echo "--- Submitting Ring Length At Most Experiments ---"
submit_job "edge_deletion/ring_length_at_most/qm9_ring_length_at_most_3_thesis.slurm" "qm9_ring_length_at_most_3_thesis"
submit_job "edge_deletion/ring_length_at_most/qm9_ring_length_at_most_4_thesis.slurm" "qm9_ring_length_at_most_4_thesis"
submit_job "edge_deletion/ring_length_at_most/qm9_ring_length_at_most_5_thesis.slurm" "qm9_ring_length_at_most_5_thesis"
submit_job "edge_deletion/ring_length_at_most/qm9_ring_length_at_most_6_thesis.slurm" "qm9_ring_length_at_most_6_thesis"

echo "=== All Jobs Submitted ==="
echo "Time: $(date)"
echo "Check job status with: squeue -u rislek" 