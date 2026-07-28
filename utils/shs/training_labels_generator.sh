#!/bin/bash
#SBATCH --output /home/usuaris/veussd/federico.costa/logs/sbatch/ser2025/%x_%j.txt
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G      # Max CPU Memory
#SBATCH --gres=gpu:0
#SBATCH --job-name=training_labels_generator

srun python '/home/usuaris/veu/federico.costa/git_repositories/SER2025/utils/training_labels_generator.py' \
	'/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/Labels/labels_consensus.csv' \
	'/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/Partitions.txt' \
	--dump_files_folder '/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/custom_data/generated_training_labels'