#!/bin/bash
#SBATCH --job-name=ctc_beto_words
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH --output=ctc_beto_words_%j.out
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --exclude=veuc01

# Re-encode the complete MrBERT cohort with BETO; keep its acoustic vectors.
# Submit the same command again to resume. No classifier training is launched.
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
dataset_root=/home/usuaris/veussd/marc.casals/datasets/WAB_samples

uv run python -u -m utils.batch_word_embeddings \
  --labels-csv "$dataset_root/labels.csv" \
  --audio-dir "$dataset_root/WAB_samples" \
  --words-dir "$dataset_root/preprocessing/words" \
  --output-root "$dataset_root/preprocessing/ctc_beto_skip_words" \
  --reuse-audio-packaged-dir "$dataset_root/preprocessing/ctc_mrbert_skip_words/embeddings" \
  --skip-unsupported-words \
  --text-model dccuchile/bert-base-spanish-wwm-uncased \
  --chunk-size 512 \
  --device cuda \
  --local-files-only \
  "$@"
