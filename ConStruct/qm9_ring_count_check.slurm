#!/bin/bash
#SBATCH --job-name=qm9_ring_count
#SBATCH --output=logs/qm9_ring_count_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000|A5000
#SBATCH --time=48:00:00
#SBATCH --partition=allgroups,testing

# Load modules and activate conda environment
source activate construct-env

# Create logs directory if it doesn't exist
mkdir -p logs

# Run ConStruct with scaled-up parameters
python ConStruct/main.py \
  --config configs/experiment/qm9.yaml \
  --config configs/general/general_default.yaml \
  model.rev_proj=ring_count \
  model.max_rings=0 \
  model.diffusion_steps=500 \
  general.number_chain_steps=200 \
  general.samples_to_generate=1000 \
  general.samples_to_save=50 \
  general.chains_to_save=10 \
  general.final_model_samples_to_generate=200 \
  general.final_model_samples_to_save=50 \
  general.final_model_chains_to_save=20 \
  train.batch_size=64 \
  train.n_epochs=1000