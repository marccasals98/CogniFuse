#!/bin/bash
#SBATCH --output /home/usuaris/veussd/federico.costa/logs/sbatch/ser2025/%x_%j.txt
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G      # Max CPU Memory
#SBATCH --gres=gpu:0
#SBATCH --job-name=generate_transcripts

srun python /home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/custom_data/generated_transcripts/v0/generate_transcripts_v0.py \
	'/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/Audios' \
	'/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/custom_data/generated_training_labels/25_01_02_17_03_37_111942/training_labels.tsv' \
	'/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/custom_data/generated_training_labels/25_01_02_17_03_37_111942/development_labels.tsv' \
	'/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/custom_data/generated_training_labels/25_01_02_17_03_37_111942/test_labels.tsv' \
	--dump_files_folder '/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/custom_data/generated_transcripts/v0' \
