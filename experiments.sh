#!/bin/bash
#SBATCH --job-name=unet_experiments
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:35:00
#SBATCH --signal=SIGUSR1@90

hostname

source $HOME/.bashrc
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load Anaconda3/2022.05
module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0
# Load the conda environment from $HOME
conda activate $HOME/.conda/envs/MedAI  

# Nodes run an A100-SXM4-40GB GPU with CUDA 12.2
# echo "Running 50_epoch_manet_resnet34_adamW_no_warmup_bce"
# Run all models 50 epochs
# Template: <Epoch>_<model>_<Depth>_<Optimizer>_<Warmup>_<Loss>
python train.py --max_epochs 50 --experiment_name 50_epoch_unet_128_adam_noWarmup_bce --model_name unet --bin 'experiments/'


