#!/bin/bash

# Script to submit all debug edge-deletion jobs
echo "=== Submitting All Debug Edge-Deletion Jobs ==="
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

# Planarity experiments
echo "--- Submitting Planarity Debug Experiments ---"
submit_job "planarity/qm9_debug_planar.slurm" "qm9_debug_planar"

# Ring count at most experiments (QM9 only)
echo "--- Submitting Ring Count At Most Debug Experiments (QM9) ---"
submit_job "ring_count_at_most/qm9_ring_count_at_most_0_debug.slurm" "qm9_ring_count_at_most_0_debug"
submit_job "ring_count_at_most/qm9_ring_count_at_most_1_debug.slurm" "qm9_ring_count_at_most_1_debug"
submit_job "ring_count_at_most/qm9_ring_count_at_most_2_debug.slurm" "qm9_ring_count_at_most_2_debug"
submit_job "ring_count_at_most/qm9_ring_count_at_most_3_debug.slurm" "qm9_ring_count_at_most_3_debug"
submit_job "ring_count_at_most/qm9_ring_count_at_most_4_debug.slurm" "qm9_ring_count_at_most_4_debug"
submit_job "ring_count_at_most/qm9_ring_count_at_most_5_debug.slurm" "qm9_ring_count_at_most_5_debug"

# Ring length at most experiments (QM9 only)
echo "--- Submitting Ring Length At Most Debug Experiments (QM9) ---"
submit_job "ring_length_at_most/qm9_ring_length_at_most_3_debug.slurm" "qm9_ring_length_at_most_3_debug"
submit_job "ring_length_at_most/qm9_ring_length_at_most_4_debug.slurm" "qm9_ring_length_at_most_4_debug"
submit_job "ring_length_at_most/qm9_ring_length_at_most_5_debug.slurm" "qm9_ring_length_at_most_5_debug"
submit_job "ring_length_at_most/qm9_ring_length_at_most_6_debug.slurm" "qm9_ring_length_at_most_6_debug"

echo "=== All Debug Jobs Submitted ==="
echo "Time: $(date)"
echo "Check job status with: squeue -u rislek"
echo "Total jobs submitted: 11 (1 planarity + 6 ring_count + 4 ring_length)" 