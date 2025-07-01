#!/bin/bash
#SBATCH --job-name=qm9_ring_length_relaxed
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/qm9_ring_length_relaxed_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000|A5000
#SBATCH --time=24:00:00
#SBATCH --partition=allgroups,testing

cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct-env


export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=/home/rislek/ConStruct-Thesis:$PYTHONPATH

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python ConStruct/main.py +experiment=qm9 \
  model.rev_proj=ring_length \
  model.max_ring_length=6 \
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