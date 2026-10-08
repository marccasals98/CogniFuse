#!/bin/bash
#SBATCH -D .
############# CHANGE THIS!!!! #######################
#SBATCH -A bsc88
#SBATCH -q acc_debug
#SBATCH --time=0-02:00:00               # Consultar batchlim para entender los límites de las particiones. 
################# CHANGE WITH YOUR RES PRIVILEGES ###########################
#SBATCH --nodes=1                       # Número de nodos
#SBATCH --ntasks=1                      # Número de tareas MPI totales
#SBATCH --ntasks-per-node=1             # Número de tareas MPI por nodo
#SBATCH --cpus-per-task=40              # Número de cores por tarea. Threads. $SLURM_CPUS_PER_TASK
#SBATCH --gres=gpu:2
################ Logging #########################
#SBATCH --job-name=skip_connection_emospeech
#SBATCH --verbose
#SBATCH --output=/gpfs/projects/bsc88/speech/speaker_recognition/outputs/ser_2025/logs/experiments_consistency/%x_%j.txt

date +%Y-%m-%d_%H:%M:%S

cd /gpfs/projects/bsc88/speech/speaker_recognition/marc_git_repositories/SER2025

# activate env
source /gpfs/projects/bsc88/speech/speaker_recognition/environments/ser_2025/bin/activate

export WANDB_MODE=offline
export WANDB_CACHE_DIR="/gpfs/projects/bsc88/speech/speaker_recognition/outputs/ser_2025/cache/wandb"
export WANDB_CONFIG_DIR="/gpfs/projects/bsc88/speech/speaker_recognition/outputs/ser_2025/cache/wandb/config"
export WANDB_DATA_DIR="/gpfs/projects/bsc88/speech/speaker_recognition/outputs/ser_2025/cache/wandb/data"
export TORCH_HOME="/gpfs/projects/bsc88/speech/speaker_recognition/outputs/ser_2025/cache/torch"
export HUGGINGFACE_HUB_CACHE="/gpfs/projects/bsc88/speech/speaker_recognition/outputs/ser_2025/cache/huggingface"
export HF_HOME="/gpfs/projects/bsc88/speech/speaker_recognition/outputs/ser_2025/cache/huggingface"

srun python scripts/train.py \
	--train_data_dir '/gpfs/projects/bsc88/speech/speaker_recognition/data/EmoSPeech2024/data/train_segments' \
	--validation_data_dir '/gpfs/projects/bsc88/speech/speaker_recognition/data/EmoSPeech2024/data/train_segments' \
	--train_labels_path '/gpfs/projects/bsc88/speech/speaker_recognition/data/EmoSPeech2024/data/train_split.tsv' \
	--validation_labels_path '/gpfs/projects/bsc88/speech/speaker_recognition/data/EmoSPeech2024/data/dev_split.tsv' \
	--dataset_transcriptions_dir '/gpfs/projects/bsc88/speech/speaker_recognition/data/EmoSPeech2024/data/transcripts' \
    --sample_rate 44100 \
	--augmentation_noises_labels_path "" \
	--augmentation_rirs_labels_path "" \
	--model_output_folder "/gpfs/projects/bsc88/speech/speaker_recognition/outputs/ser_2025/models" \
	--log_file_folder "/gpfs/projects/bsc88/speech/speaker_recognition/outputs/ser_2025/logs/train" \
	--wandb_dir "/gpfs/projects/bsc88/speech/speaker_recognition/outputs/ser_2025/wandb" \
	--training_random_crop_secs 5.5 \
	--evaluation_random_crop_secs 0 \
	--augmentation_window_size_secs 5.5 \
	--training_augmentation_prob 0 \
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
	--max_epochs 20 \
	--training_batch_size 32 \
	--evaluation_batch_size 1 \
	--eval_and_save_best_model_every 100 \
	--print_training_info_every 100 \
	--early_stopping 0 \
	--num_workers 4 \
	--padding_type 'repetition_pad' \
	--classifier_hidden_layers 4 \
	--classifier_hidden_layers_width 512 \
	--classifier_layer_drop_out 0.1 \
	--number_classes 6 \
	--loss 'CrossEntropy' \
	--weighted_loss \
	--optimizer 'adamw' \
	--update_optimizer_every 2 \
	--learning_rate 0.0001 \
	--learning_rate_multiplier 0.5 \
	--weight_decay 0.01 \
	--use_weights_and_biases \
	--skip_connections 


date +%Y-%m-%d_%H:%M:%S