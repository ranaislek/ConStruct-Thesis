#!/bin/bash
#SBATCH --job-name=qm9_planar_comp
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/qm9_planar_comp_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000|A5000
#SBATCH --time=48:00:00
#SBATCH --partition=allgroups,testing

cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=/home/rislek/ConStruct-Thesis:$PYTHONPATH

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Run ConStruct with planarity constraint for comparison
python ConStruct/main.py +experiment=qm9 \
  model.rev_proj=planar \
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