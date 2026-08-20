#!/bin/bash
#SBATCH --output /home/usuaris/veussd/marc.casals/logs/sbatch/ser2025/%x_%j.txt
#SBATCH -A veu
#SBATCH -p veu            # Partition to submit to
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --job-name=train

date

# NCCL Stability Fix (Prevents timeouts on nodes like veuc12)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0


# Use torchrun with uv for distributed data parallel training
# --nproc_per_node should match the number of GPUs requested (#SBATCH --gres=gpu:2)
uv run torchrun --nproc_per_node=4 scripts/train.py \
	--train_data_dir '/home/usuaris/veussd/marc.casals/datasets/WAB_samples/audios' \
	--validation_data_dir '/home/usuaris/veussd/marc.casals/datasets/WAB_samples/audios' \
	--train_labels_path '/home/usuaris/veussd/marc.casals/datasets/WAB_samples/labels.csv' \
	--validation_labels_path '/home/usuaris/veussd/marc.casals/datasets/WAB_samples/labels.csv' \
	--augmentation_noises_labels_path "/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/custom_data/generated_augmentation_labels/data_augmentation_noises_labels.tsv" \
	--augmentation_rirs_labels_path "/home/usuaris/veussd/federico.costa/datasets/msp_podcast_2025/custom_data/generated_augmentation_labels/data_augmentation_rirs_labels.tsv" \
	--model_output_folder "/home/usuaris/veussd/marc.casals/models" \
	--log_file_folder "/home/usuaris/veussd/marc.casals/logs/cognifuse/train" \
	--training_random_crop_secs 5.5 \
	--evaluation_random_crop_secs 0 \
	--augmentation_window_size_secs 5.5 \
	--training_augmentation_prob 0.5 \
	--evaluation_augmentation_prob 0 \
	--augmentation_effects 'apply_speed_perturbation' 'apply_reverb' 'add_background_noise' \
	--speech_feature_extractor 'WAV2VEC2_XLSR_300M' \
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
	--training_batch_size 1\
	--evaluation_batch_size 1 \
	--eval_and_save_best_model_every 1600 \
	--print_training_info_every 100 \
	--early_stopping 0 \
	--num_workers 0 \
	--padding_type 'repetition_pad' \
	--classifier_hidden_layers 4 \
	--classifier_hidden_layers_width 512 \
	--classifier_layer_drop_out 0.1 \
	--number_classes 3 \
	--loss 'CrossEntropy' \
	--weighted_loss \
	--optimizer 'adamw' \
	--update_optimizer_every 2 \
	--learning_rate 0.0001 \
	--learning_rate_multiplier 0.5 \
	--weight_decay 0.01 \
	--window_secs 40\
	--stride_secs 7.0\
	--num_folds 5\
	--whisper_model_name tiny\
	--whisper_language es\
	--use_weights_and_biases

date
