#!/bin/bash
#SBATCH --job-name=adresso_baseline_cv
#SBATCH --output=/home/usuaris/veussd/marc.casals/logs/cognifuse/train/slurm-%j.log
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --exclude=veuc01

set -euo pipefail

export ADRESSO_FEATURES_ROOT="/home/usuaris/veu/marc.casals/CogniFuse/outputs/adresso_ctc_distilbert_words_20261029"


features_root="${ADRESSO_FEATURES_ROOT:?Set ADRESSO_FEATURES_ROOT to the existing CTC feature export}"
run_root="${ADRESSO_CV_RUN_ROOT:-/home/usuaris/veussd/marc.casals/logs/cognifuse/train/run_${SLURM_JOB_ID:-manual}}"



export TMPDIR=/tmp
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# One GPU; the CV runner starts an independent process for each saved fold.
# Only the 166 labeled training recordings participate in development CV.
uv run --no-sync python -u scripts/train.py \
  --dataset adresso \
  --dataset_root /home/usuaris/veussd/marc.casals/datasets/ADReSSo21 \
  --precomputed_features_dir "$features_root/train/embeddings" \
  --precomputed_text_suffix _word_text.pt \
  --precomputed_audio_suffix _word_audio.pt \
  --model_output_folder "$run_root/models" \
  --log_file_folder "$run_root/logs" \
  --cross_validation_output_dir "$run_root/results" \
  --cross_validate --num_folds 5 --number_classes 2 \
  --random_seed 1234 --torch_seed 3407 \
  --speech_adapter NoneAdapter --text_adapter NoneAdapter \
  --no-skip_connections \
  --seq_to_seq_method MultiHeadAttention --seq_to_seq_heads_number 4 \
  --seq_to_seq_input_dropout 0.0 \
  --seq_to_one_method AttentionPooling --seq_to_one_input_dropout 0.0 \
  --classifier_hidden_layers 4 --classifier_hidden_layers_width 512 \
  --classifier_layer_drop_out 0.1 \
  --max_epochs 60 --training_batch_size 1 --evaluation_batch_size 1 \
  --training_augmentation_prob 0 --evaluation_augmentation_prob 0 \
  --num_workers 0 \
  --eval_and_save_best_model_every 100 --print_training_info_every 100 \
  --early_stopping 0 --update_optimizer_every 0 \
  --loss CrossEntropy --weighted_loss \
  --optimizer adamw --learning_rate 0.0001 --weight_decay 0.01 \
  --no-load_checkpoint --no-use_weights_and_biases \
  "$@"
