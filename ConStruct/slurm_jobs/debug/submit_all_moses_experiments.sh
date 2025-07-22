#!/bin/bash

# ALL MOSES Experiments Submission Script
# This script submits ALL MOSES debug experiments to comprehensively test
# both edge-insertion and edge-deletion mechanisms with larger molecules (25+ nodes)

echo "=== Submitting ALL MOSES Debug Experiments ==="
echo "Purpose: Comprehensive testing of both edge-insertion and edge-deletion with MOSES dataset"
echo "Dataset: MOSES (larger molecules, 25+ nodes, better for complex constraints)"
echo "Experiments: Edge-Insertion, Edge-Deletion, No-Constraint"
echo "Time: $(date)"
echo ""

# Change to the slurm jobs directory
cd /home/rislek/ConStruct-Thesis/ConStruct/slurm_jobs/debug

# Submit MOSES no-constraint experiment (baseline)
echo "Submitting MOSES no-constraint debug (baseline)..."
job_no_constraint=$(sbatch no_constraint/moses_no_constraint_debug.slurm)
echo "Job ID: $job_no_constraint"

# Submit MOSES edge-insertion experiments
echo ""
echo "Submitting MOSES edge-insertion experiments..."
echo "Submitting MOSES ring_count_at_least experiments..."
job_ring_count_least_1=$(sbatch edge_insertion/ring_count_at_least/moses_ring_count_at_least_1_debug.slurm)
echo "Job ID: $job_ring_count_least_1"

job_ring_count_least_2=$(sbatch edge_insertion/ring_count_at_least/moses_ring_count_at_least_2_debug.slurm)
echo "Job ID: $job_ring_count_least_2"

job_ring_count_least_3=$(sbatch edge_insertion/ring_count_at_least/moses_ring_count_at_least_3_debug.slurm)
echo "Job ID: $job_ring_count_least_3"

echo "Submitting MOSES ring_length_at_least experiments..."
job_ring_length_least_4=$(sbatch edge_insertion/ring_length_at_least/moses_ring_length_at_least_4_debug.slurm)
echo "Job ID: $job_ring_length_least_4"

job_ring_length_least_5=$(sbatch edge_insertion/ring_length_at_least/moses_ring_length_at_least_5_debug.slurm)
echo "Job ID: $job_ring_length_least_5"

job_ring_length_least_6=$(sbatch edge_insertion/ring_length_at_least/moses_ring_length_at_least_6_debug.slurm)
echo "Job ID: $job_ring_length_least_6"

# Submit MOSES edge-deletion experiments
echo ""
echo "Submitting MOSES edge-deletion experiments..."
echo "Submitting MOSES ring_count_at_most experiments..."
job_ring_count_most_2=$(sbatch edge_deletion/ring_count_at_most/moses_ring_count_at_most_2_debug.slurm)
echo "Job ID: $job_ring_count_most_2"

job_ring_count_most_3=$(sbatch edge_deletion/ring_count_at_most/moses_ring_count_at_most_3_debug.slurm)
echo "Job ID: $job_ring_count_most_3"

job_ring_count_most_4=$(sbatch edge_deletion/ring_count_at_most/moses_ring_count_at_most_4_debug.slurm)
echo "Job ID: $job_ring_count_most_4"

echo "Submitting MOSES ring_length_at_most experiments..."
job_ring_length_most_4=$(sbatch edge_deletion/ring_length_at_most/moses_ring_length_at_most_4_debug.slurm)
echo "Job ID: $job_ring_length_most_4"

job_ring_length_most_5=$(sbatch edge_deletion/ring_length_at_most/moses_ring_length_at_most_5_debug.slurm)
echo "Job ID: $job_ring_length_most_5"

job_ring_length_most_6=$(sbatch edge_deletion/ring_length_at_most/moses_ring_length_at_most_6_debug.slurm)
echo "Job ID: $job_ring_length_most_6"

echo ""
echo "=== ALL MOSES Debug Jobs Submitted ==="
echo ""
echo "No-Constraint (baseline): $job_no_constraint"
echo ""
echo "Edge-Insertion Experiments:"
echo "  Ring Count At Least 1: $job_ring_count_least_1"
echo "  Ring Count At Least 2: $job_ring_count_least_2"
echo "  Ring Count At Least 3: $job_ring_count_least_3"
echo "  Ring Length At Least 4: $job_ring_length_least_4"
echo "  Ring Length At Least 5: $job_ring_length_least_5"
echo "  Ring Length At Least 6: $job_ring_length_least_6"
echo ""
echo "Edge-Deletion Experiments:"
echo "  Ring Count At Most 2: $job_ring_count_most_2"
echo "  Ring Count At Most 3: $job_ring_count_most_3"
echo "  Ring Count At Most 4: $job_ring_count_most_4"
echo "  Ring Length At Most 4: $job_ring_length_most_4"
echo "  Ring Length At Most 5: $job_ring_length_most_5"
echo "  Ring Length At Most 6: $job_ring_length_most_6"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in: /home/rislek/ConStruct-Thesis/ConStruct/logs/"
echo ""
echo "Expected Results Analysis:"
echo "- MOSES molecules should have larger graphs (25+ nodes)"
echo "- Edge-insertion should work better with larger molecules"
echo "- Edge-deletion should work on complex graphs"
echo "- Compare edge-insertion vs edge-deletion performance"
echo "- Compare with QM9 results to see dataset differences"
echo ""
echo "ConStruct Philosophy Analysis:"
echo "- Structural constraints enforced during generation"
echo "- Chemical properties measured post-generation"
echo "- Gap analysis enables honest, critical thesis results"
echo "- Systematic comparison of transition mechanisms"
echo ""
echo "Thesis Implications:"
echo "- Honest evaluation of constraint satisfaction"
echo "- Systematic comparison of edge-insertion vs edge-deletion"
echo "- Analysis of structural vs chemical validity gap"
echo "- Practical relevance with drug-like molecules"
echo ""
echo "Submission completed at $(date)" 