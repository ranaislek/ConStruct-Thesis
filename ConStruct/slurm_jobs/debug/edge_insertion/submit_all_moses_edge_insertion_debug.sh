#!/bin/bash

# MOSES Edge-Insertion Debug Experiments Submission Script
# This script submits all MOSES edge-insertion debug experiments to test
# the edge-insertion mechanism with larger molecules (25+ nodes)

echo "=== Submitting All MOSES Edge-Insertion Debug Experiments ==="
echo "Purpose: Test edge-insertion constraints with MOSES dataset"
echo "Dataset: MOSES (larger molecules, 25+ nodes, better for edge-insertion)"
echo "Constraints: ring_count_at_least, ring_length_at_least, no_constraint"
echo "Transition: edge_insertion"
echo "Time: $(date)"
echo ""

# Change to the slurm jobs directory
cd /home/rislek/ConStruct-Thesis/ConStruct/slurm_jobs/debug

# Submit MOSES no-constraint experiment (baseline)
echo "Submitting MOSES no-constraint debug..."
job_no_constraint=$(sbatch no_constraint/moses_no_constraint_debug.slurm)
echo "Job ID: $job_no_constraint"

# Submit MOSES ring count at least experiments
echo ""
echo "Submitting MOSES ring_count_at_least experiments..."
echo "Submitting MOSES ring_count_at_least_1_debug..."
job_ring_count_1=$(sbatch edge_insertion/ring_count_at_least/moses_ring_count_at_least_1_debug.slurm)
echo "Job ID: $job_ring_count_1"

echo "Submitting MOSES ring_count_at_least_2_debug..."
job_ring_count_2=$(sbatch edge_insertion/ring_count_at_least/moses_ring_count_at_least_2_debug.slurm)
echo "Job ID: $job_ring_count_2"

echo "Submitting MOSES ring_count_at_least_3_debug..."
job_ring_count_3=$(sbatch edge_insertion/ring_count_at_least/moses_ring_count_at_least_3_debug.slurm)
echo "Job ID: $job_ring_count_3"

# Submit MOSES ring length at least experiments
echo ""
echo "Submitting MOSES ring_length_at_least experiments..."
echo "Submitting MOSES ring_length_at_least_4_debug..."
job_ring_length_4=$(sbatch edge_insertion/ring_length_at_least/moses_ring_length_at_least_4_debug.slurm)
echo "Job ID: $job_ring_length_4"

echo "Submitting MOSES ring_length_at_least_5_debug..."
job_ring_length_5=$(sbatch edge_insertion/ring_length_at_least/moses_ring_length_at_least_5_debug.slurm)
echo "Job ID: $job_ring_length_5"

echo "Submitting MOSES ring_length_at_least_6_debug..."
job_ring_length_6=$(sbatch edge_insertion/ring_length_at_least/moses_ring_length_at_least_6_debug.slurm)
echo "Job ID: $job_ring_length_6"

echo ""
echo "=== All MOSES Edge-Insertion Debug Jobs Submitted ==="
echo "No-Constraint (baseline): $job_no_constraint"
echo "Ring Count At Least 1: $job_ring_count_1"
echo "Ring Count At Least 2: $job_ring_count_2"
echo "Ring Count At Least 3: $job_ring_count_3"
echo "Ring Length At Least 4: $job_ring_length_4"
echo "Ring Length At Least 5: $job_ring_length_5"
echo "Ring Length At Least 6: $job_ring_length_6"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in: /home/rislek/ConStruct-Thesis/ConStruct/logs/"
echo ""
echo "Expected Results:"
echo "- MOSES molecules should have larger graphs (25+ nodes)"
echo "- Edge-insertion should work better with larger molecules"
echo "- Compare with QM9 results to see improvement"
echo "- No-constraint baseline for comparison"
echo ""
echo "ConStruct Philosophy Analysis:"
echo "- Structural constraints enforced during generation"
echo "- Chemical properties measured post-generation"
echo "- Gap analysis enables honest, critical thesis results"
echo ""
echo "Submission completed at $(date)" 