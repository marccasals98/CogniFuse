#!/bin/bash
#SBATCH --output /home/usuaris/veussd/federico.costa/logs/sbatch/ser2025/%x_%j.txt
#SBATCH -A veu 
#SBATCH -p veu            # Partition to submit to
#SBATCH --cpus-per-task=20
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --job-name=train

date

srun python scripts/train.py \
	--train_data_dir '/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/Audios' \
	--validation_data_dir '/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/Audios' \
	--train_labels_path '/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/custom_data/generated_training_labels/25_01_02_17_03_37_111942/training_labels.tsv' \
	--validation_labels_path '/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/custom_data/generated_training_labels/25_01_02_17_03_37_111942/development_labels.tsv' \
	--dataset_transcriptions_dir '/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/custom_data/generated_transcripts/v1/25_01_04_13_38_12_603360/transcripts' \
	--augmentation_noises_labels_path "/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/custom_data/generated_augmentation_labels/data_augmentation_noises_labels.tsv" \
	--augmentation_rirs_labels_path "/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/custom_data/generated_augmentation_labels/data_augmentation_rirs_labels.tsv" \
	--model_output_folder "/home/usuaris/veussd/federico.costa/models/" \
	--log_file_folder "/home/usuaris/veussd/federico.costa/logs/train/" \
	--training_random_crop_secs 5.5 \
	--evaluation_random_crop_secs 0 \
	--augmentation_window_size_secs 5.5 \
	--training_augmentation_prob 0 \
	--evaluation_augmentation_prob 0 \
	--speech_feature_extractor 'WAVLM_LARGE' \
	--speech_feature_extractor_output_vectors_dimension 1024 \
	--text_feature_extractor 'BERT_LARGE_UNCASED' \
	--text_feature_extractor_output_vectors_dimension 1024 \
	--speech_adapter 'NoneAdapter' \
	--text_adapter 'NoneAdapter' \
	--seq_to_seq_method 'MultiHeadAttention' \
	--seq_to_seq_heads_number 4 \
	--seq_to_seq_input_dropout 0.0 \
	--seq_to_one_method 'AttentionPooling' \
	--seq_to_one_input_dropout 0.0 \
	--max_epochs 10 \
	--training_batch_size 32 \
	--evaluation_batch_size 1 \
	--eval_and_save_best_model_every 1600 \
	--print_training_info_every 100 \
	--early_stopping 0 \
	--num_workers 4 \
	--padding_type 'repetition_pad' \
	--classifier_hidden_layers 4 \
	--classifier_hidden_layers_width 512 \
	--classifier_layer_drop_out 0.1 \
	--number_classes 8 \
	--loss 'FocalLoss' \
	--weighted_loss \
	--optimizer 'adamw' \
	--update_optimizer_every 2 \
	--learning_rate 0.0001 \
	--learning_rate_multiplier 0.5 \
	--weight_decay 0.01 \
	--use_weights_and_biases

date