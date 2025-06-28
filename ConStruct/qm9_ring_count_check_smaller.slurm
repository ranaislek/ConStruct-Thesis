#!/bin/bash
#SBATCH --job-name=qm9_ring_count_small
#SBATCH --output=logs/qm9_ring_count_small_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000|A5000
#SBATCH --time=24:00:00
#SBATCH --partition=allgroups,testing

# Activate conda environment
source activate construct-env

# Create logs directory if it doesn't exist
mkdir -p logs

# Run ConStruct with moderate parameters
python ConStruct/main.py \
  --config configs/experiment/qm9.yaml \
  --config configs/general/general_default.yaml \
  model.rev_proj=ring_count \
  model.max_rings=0 \
  model.diffusion_steps=200 \
  general.number_chain_steps=50 \
  general.samples_to_generate=200 \
  general.samples_to_save=20 \
  general.chains_to_save=5 \
  general.final_model_samples_to_generate=50 \
  general.final_model_samples_to_save=20 \
  general.final_model_chains_to_save=10 \
  train.batch_size=32 \
  train.n_epochs=200
