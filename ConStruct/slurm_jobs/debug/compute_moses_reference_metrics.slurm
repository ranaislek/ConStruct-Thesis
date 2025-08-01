#!/bin/bash
#SBATCH --job-name=compute_moses_ref_metrics
#SBATCH --output=/home/rislek/ConStruct-Thesis/ConStruct/logs/compute_moses_ref_metrics_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000|A5000
#SBATCH --time=24:00:00
#SBATCH --partition=allgroups,testing

# Load environment
cd /home/rislek/ConStruct-Thesis
source /conf/shared-software/anaconda/etc/profile.d/conda.sh
conda activate construct-env

export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"

# WandB fixes to prevent hanging
export WANDB_TIMEOUT=300
export WANDB_START_METHOD=thread
export WANDB_CONSOLE=wrap
export WANDB_SILENT=false
export WANDB_NETWORK_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=300

# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "=== Computing MOSES Reference Metrics ==="
echo "Purpose: Compute and save reference metrics for MOSES dataset"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

# Run a minimal experiment just to compute reference metrics
cd ConStruct
python -c "
from ConStruct.datasets.moses_dataset import MosesDataModule, MosesInfos
from ConStruct.datasets.dataset_utils import compute_reference_metrics
import hydra
from omegaconf import DictConfig

# Create minimal config
class MinimalConfig:
    def __init__(self):
        self.dataset = type('obj', (object,), {
            'datadir': 'data/moses',
            'remove_h': False,
            'batch_size': 128,
            'num_workers': 4
        })()
        self.general = type('obj', (object,), {
            'guidance_target': None
        })()

cfg = MinimalConfig()

# Initialize data module
datamodule = MosesDataModule(cfg)
datamodule.prepare_dataloader()

# Initialize dataset infos
dataset_infos = MosesInfos(datamodule, cfg)

# Compute reference metrics
print('Computing reference metrics...')
compute_reference_metrics(dataset_infos, datamodule)

print('Reference metrics computed and saved successfully!')
"

echo "Reference metrics computation completed at $(date)" 