#!/bin/bash
#SBATCH --job-name=guacamol_dataset
#SBATCH --output=ConStruct/logs/guacamol_dataset_processing_%j.out
#SBATCH --error=ConStruct/logs/guacamol_dataset_processing_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1

echo "=== Guacamol Dataset Processing Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate construct-env

# Set up environment
cd /home/rislek/ConStruct-Thesis

echo "=== Processing Full Guacamol Dataset ==="
echo "Purpose: Download and process complete Guacamol dataset"
echo "Dataset: Guacamol (large molecules, proper charge information)"
echo "Processing: Full dataset (~238,704 molecules total)"
echo ""

# Remove old processed files to ensure clean processing
echo "Cleaning old processed files..."
rm -rf /extra/rislek/ConStruct-Thesis/data/guacamol/processed/*

# Run the dataset processing
echo "Starting Guacamol dataset processing..."
python setup_guacamol_dataset.py

# Verify the results
echo ""
echo "=== Verification ==="
echo "Checking processed files..."
ls -la /extra/rislek/ConStruct-Thesis/data/guacamol/processed/

echo ""
echo "Checking file sizes..."
du -h /extra/rislek/ConStruct-Thesis/data/guacamol/processed/*.pt

echo ""
echo "=== Guacamol Dataset Processing Complete ==="
echo "Job completed at: $(date)"
echo "Dataset ready for thesis experiments" 