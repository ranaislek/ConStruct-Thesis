#!/bin/bash

# MOSES Edge-Insertion Debug Experiments Submission Script
# This script submits all MOSES edge-insertion debug experiments to test
# the edge-insertion mechanism with larger molecules (25+ nodes)

echo "=== Submitting MOSES Edge-Insertion Debug Experiments ==="
echo "Purpose: Test edge-insertion constraints with MOSES dataset"
echo "Dataset: MOSES (larger molecules, 25+ nodes, better for edge-insertion)"
echo "Constraints: ring_count_at_least (1, 2, 3 rings)"
echo "Transition: edge_insertion"
echo "Time: $(date)"
echo ""

# Change to the slurm jobs directory
cd /home/rislek/ConStruct-Thesis/ConStruct/slurm_jobs/debug/edge_insertion/ring_count_at_least

# Submit MOSES edge-insertion experiments
echo "Submitting MOSES ring_count_at_least_1_debug..."
job1=$(sbatch moses_ring_count_at_least_1_debug.slurm)
echo "Job ID: $job1"

echo "Submitting MOSES ring_count_at_least_2_debug..."
job2=$(sbatch moses_ring_count_at_least_2_debug.slurm)
echo "Job ID: $job2"

echo "Submitting MOSES ring_count_at_least_3_debug..."
job3=$(sbatch moses_ring_count_at_least_3_debug.slurm)
echo "Job ID: $job3"

echo ""
echo "=== All MOSES Edge-Insertion Debug Jobs Submitted ==="
echo "Job 1 (1 ring): $job1"
echo "Job 2 (2 rings): $job2"
echo "Job 3 (3 rings): $job3"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in: /home/rislek/ConStruct-Thesis/ConStruct/logs/"
echo ""
echo "Expected Results:"
echo "- MOSES molecules should have larger graphs (25+ nodes)"
echo "- Edge-insertion should work better with larger molecules"
echo "- Compare with QM9 results to see improvement"
echo ""
echo "Submission completed at $(date)" 