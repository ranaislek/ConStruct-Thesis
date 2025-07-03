#!/bin/bash
#SBATCH --job-name=qm9_ring_count_real
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/qm9_ring_count_real_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000|A5000
#SBATCH --time=24:00:00
#SBATCH --partition=allgroups,testing

cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

python -c 'import torch; print("torch version:", torch.__version__); print("CUDA version:", torch.version.cuda)'
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("CUDA device count:", torch.cuda.device_count())'
nvidia-smi
which nvcc && nvcc --version

python ConStruct/main.py +experiment=qm9 \
  model.rev_proj=ring_count \
  model.max_rings=0 \
  model.diffusion_steps=1000 \
  general.number_chain_steps=100 \
  general.samples_to_generate=500 \
  general.samples_to_save=50 \
  general.chains_to_save=10 \
  general.final_model_samples_to_generate=100 \
  general.final_model_samples_to_save=50 \
  general.final_model_chains_to_save=20 \
  train.batch_size=32 \
  train.n_epochs=200 