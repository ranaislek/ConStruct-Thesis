#!/bin/bash

# ConStruct SLURM Job Submission Script
# Usage: ./submit_experiments.sh [debug|real|test] [ring_count|ring_length|both]

if [ $# -lt 2 ]; then
    echo "Usage: $0 [debug|real|test] [ring_count|ring_length|both]"
    echo ""
    echo "Examples:"
    echo "  $0 debug ring_count     # Submit debug test for ring count"
    echo "  $0 real ring_length     # Submit real experiment for ring length"
    echo "  $0 test both            # Submit validation test for both constraints"
    exit 1
fi

EXPERIMENT_TYPE=$1
CONSTRAINT_TYPE=$2

case $EXPERIMENT_TYPE in
    "debug")
        case $CONSTRAINT_TYPE in
            "ring_count")
                echo "Submitting debug test for ring count constraint..."
                sbatch ConStruct/slurm_jobs/debug/qm9_ring_count_debug.slurm
                ;;
            "ring_length")
                echo "Submitting debug test for ring length constraint..."
                sbatch ConStruct/slurm_jobs/debug/qm9_ring_length_debug.slurm
                ;;
            *)
                echo "Invalid constraint type for debug: $CONSTRAINT_TYPE"
                echo "Valid options: ring_count, ring_length"
                exit 1
                ;;
        esac
        ;;
    "real")
        case $CONSTRAINT_TYPE in
            "ring_count")
                echo "Submitting real experiment for ring count constraint..."
                sbatch ConStruct/slurm_jobs/experiments/qm9_ring_count_real.slurm
                ;;
            "ring_length")
                echo "Submitting real experiment for ring length constraint..."
                sbatch ConStruct/slurm_jobs/experiments/qm9_ring_length_real.slurm
                ;;
            *)
                echo "Invalid constraint type for real experiment: $CONSTRAINT_TYPE"
                echo "Valid options: ring_count, ring_length"
                exit 1
                ;;
        esac
        ;;
    "test")
        case $CONSTRAINT_TYPE in
            "both")
                echo "Submitting validation test for both constraints..."
                sbatch ConStruct/slurm_jobs/tests/qm9_constraint_validation.slurm
                ;;
            *)
                echo "Invalid constraint type for test: $CONSTRAINT_TYPE"
                echo "Valid options: both"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Invalid experiment type: $EXPERIMENT_TYPE"
        echo "Valid options: debug, real, test"
        exit 1
        ;;
esac

echo "Job submitted successfully!"
echo "Check status with: squeue -u rislek" 