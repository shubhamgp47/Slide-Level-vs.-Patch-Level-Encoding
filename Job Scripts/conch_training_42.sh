#!/bin/bash -l
#SBATCH --job-name=conch_training_seed42
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
source ~/.bashrc
module load python
conda activate histgen
bash /home/woody/iwi5/iwi5204h/HistGen/After_Thesis/train_wsi_report_conch_histgen_seed42.sh