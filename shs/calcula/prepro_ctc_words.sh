#!/bin/bash
#SBATCH --job-name=ctc_mrbert_words
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH --output=ctc_mrbert_words_%j.out
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --exclude=veuc01

# Resume by submitting the same command again. No training is launched here.
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
dataset_root=/home/usuaris/veussd/marc.casals/datasets/WAB_samples

uv run python -u -m utils.batch_word_embeddings \
  --labels-csv "$dataset_root/labels.csv" \
  --audio-dir "$dataset_root/WAB_samples" \
  --words-dir "$dataset_root/preprocessing/words" \
  --output-root "$dataset_root/preprocessing/ctc_mrbert_batch" \
  --reuse-packaged-dir "$dataset_root/preprocessing/word_embeddings/step9_first_recording" \
  --text-model BSC-LT/MrBERT \
  --device cuda \
  --local-files-only \
  "$@"
