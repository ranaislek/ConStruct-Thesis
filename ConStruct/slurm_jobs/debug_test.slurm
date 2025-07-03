#!/bin/bash
#SBATCH --job-name=construct_debug_qm9
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/debug_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --partition=debug
#SBATCH --time=01:00:00    # 10 minutes were too less

cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'

python ConStruct/main.py +experiment=qm9 train.n_epochs=1 train.batch_size=8 general.samples_to_generate=8 general.number_chain_steps=1