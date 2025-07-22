#!/bin/bash

# MOSES Edge-Deletion Debug Experiments Submission Script
# This script submits all MOSES edge-deletion debug experiments to test
# the edge-deletion mechanism with larger molecules (25+ nodes)

echo "=== Submitting All MOSES Edge-Deletion Debug Experiments ==="
echo "Purpose: Test edge-deletion constraints with MOSES dataset"
echo "Dataset: MOSES (larger molecules, 25+ nodes, test edge-deletion on complex graphs)"
echo "Constraints: ring_count_at_most, ring_length_at_most"
echo "Transition: absorbing_edges (edge-deletion)"
echo "Time: $(date)"
echo ""

# Change to the slurm jobs directory
cd /home/rislek/ConStruct-Thesis/ConStruct/slurm_jobs/debug

# Submit MOSES ring count at most experiments
echo "Submitting MOSES ring_count_at_most experiments..."
echo "Submitting MOSES ring_count_at_most_2_debug..."
job_ring_count_2=$(sbatch edge_deletion/ring_count_at_most/moses_ring_count_at_most_2_debug.slurm)
echo "Job ID: $job_ring_count_2"

echo "Submitting MOSES ring_count_at_most_3_debug..."
job_ring_count_3=$(sbatch edge_deletion/ring_count_at_most/moses_ring_count_at_most_3_debug.slurm)
echo "Job ID: $job_ring_count_3"

echo "Submitting MOSES ring_count_at_most_4_debug..."
job_ring_count_4=$(sbatch edge_deletion/ring_count_at_most/moses_ring_count_at_most_4_debug.slurm)
echo "Job ID: $job_ring_count_4"

# Submit MOSES ring length at most experiments
echo ""
echo "Submitting MOSES ring_length_at_most experiments..."
echo "Submitting MOSES ring_length_at_most_4_debug..."
job_ring_length_4=$(sbatch edge_deletion/ring_length_at_most/moses_ring_length_at_most_4_debug.slurm)
echo "Job ID: $job_ring_length_4"

echo "Submitting MOSES ring_length_at_most_5_debug..."
job_ring_length_5=$(sbatch edge_deletion/ring_length_at_most/moses_ring_length_at_most_5_debug.slurm)
echo "Job ID: $job_ring_length_5"

echo "Submitting MOSES ring_length_at_most_6_debug..."
job_ring_length_6=$(sbatch edge_deletion/ring_length_at_most/moses_ring_length_at_most_6_debug.slurm)
echo "Job ID: $job_ring_length_6"

echo ""
echo "=== All MOSES Edge-Deletion Debug Jobs Submitted ==="
echo "Ring Count At Most 2: $job_ring_count_2"
echo "Ring Count At Most 3: $job_ring_count_3"
echo "Ring Count At Most 4: $job_ring_count_4"
echo "Ring Length At Most 4: $job_ring_length_4"
echo "Ring Length At Most 5: $job_ring_length_5"
echo "Ring Length At Most 6: $job_ring_length_6"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in: /home/rislek/ConStruct-Thesis/ConStruct/logs/"
echo ""
echo "Expected Results:"
echo "- MOSES molecules should have larger graphs (25+ nodes)"
echo "- Edge-deletion should work on complex graphs"
echo "- Compare with QM9 results to see performance differences"
echo "- Compare with edge-insertion results for thesis analysis"
echo ""
echo "ConStruct Philosophy Analysis:"
echo "- Structural constraints enforced during generation"
echo "- Chemical properties measured post-generation"
echo "- Gap analysis enables honest, critical thesis results"
echo ""
echo "Submission completed at $(date)" 