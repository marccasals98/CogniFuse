#!/bin/bash
#SBATCH --output /home/usuaris/veussd/federico.costa/logs/sbatch/ser2025/%x_%j.txt
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G      # Max CPU Memory
#SBATCH --gres=gpu:0
#SBATCH --job-name=data_augmentation_labels_generator

srun python /home/usuaris/veu/federico.costa/git_repositories/SER2025/utils/data_augmentation_labels_generator.py \
	--rirs_data_folders "/home/usuaris/scratch/speaker_databases/RIRS_NOISES/real_rirs_isotropic_noises/" "/home/usuaris/scratch/speaker_databases/RIRS_NOISES/simulated_rirs/" \
	--noises_data_folders "/home/usuaris/scratch/speaker_databases/RIRS_NOISES/pointsource_noises/" "/home/usuaris/scratch/speaker_databases/musan/music/" "/home/usuaris/scratch/speaker_databases/musan/noise/" "/home/usuaris/scratch/speaker_databases/musan/speech" \
	--dump_files_folder '/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/custom_data/generated_training_labels'