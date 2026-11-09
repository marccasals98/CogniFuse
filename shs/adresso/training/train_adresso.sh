#!/bin/bash
#SBATCH --job-name=adresso_fold1
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH --output=/home/usuaris/veu/marc.casals/CogniFuse/logs/adresso_train_%j.out
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --exclude=veuc01

set -euo pipefail

repo_root=/home/usuaris/veu/marc.casals/CogniFuse
dataset_root=/home/usuaris/veussd/marc.casals/datasets/ADReSSo21
features_dir="$repo_root/outputs/adresso_full_whisper_distil_pauses/train/embeddings"
training_output="$repo_root/outputs/adresso_training"
cd "$repo_root"

# Keep the cluster settings used by the existing training launcher.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_MNNVL_ENABLE=0
export NCCL_SHM_DISABLE=1
export NCCL_DMABUF_ENABLE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export TMPDIR=/tmp

date

# Reuse the current attention/head configuration with prepared ADReSSo pairs.
# Feature encoders and waveform augmentation are skipped in this workflow.
# Extra arguments override these defaults, e.g. --fold 0 or --cross_validate.
uv run --no-sync torchrun --nproc_per_node=1 --standalone scripts/train.py \
  --dataset adresso \
  --dataset_root "$dataset_root" \
  --precomputed_features_dir "$features_dir" \
  --precomputed_text_suffix distil_pauses.pt \
  --precomputed_audio_suffix distil_pauses_audio.pt \
  --model_output_folder "$training_output/models" \
  --log_file_folder "$training_output/logs" \
  --number_classes 2 \
  --num_folds 5 \
  --fold 1 \
  --speech_adapter NoneAdapter \
  --text_adapter NoneAdapter \
  --seq_to_seq_method MultiHeadAttention \
  --seq_to_seq_heads_number 4 \
  --seq_to_seq_input_dropout 0.0 \
  --seq_to_one_method AttentionPooling \
  --seq_to_one_input_dropout 0.0 \
  --classifier_hidden_layers 4 \
  --classifier_hidden_layers_width 512 \
  --classifier_layer_drop_out 0.1 \
  --max_epochs 60 \
  --training_batch_size 1 \
  --evaluation_batch_size 1 \
  --num_workers 0 \
  --eval_and_save_best_model_every 100 \
  --print_training_info_every 100 \
  --early_stopping 0 \
  --loss CrossEntropy \
  --weighted_loss \
  --optimizer adamw \
  --learning_rate 0.0001 \
  --learning_rate_multiplier 0.5 \
  --weight_decay 0.01 \
  --update_optimizer_every 2 \
  --no-load_checkpoint \
  --no-use_weights_and_biases \
  "$@"

date
