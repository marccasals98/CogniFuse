#!/bin/bash
#SBATCH --job-name=ctc_mrbert_skip
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH --output=ctc_mrbert_skip_%j.out
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --exclude=veuc01

# Keep the strict batch intact; reuse its 172 packaged recordings by validation.
# This reserves two GPUs; the current worker still places each encoder on one GPU.
set -euo pipefail
dataset_root=/home/usuaris/veussd/marc.casals/datasets/WAB_samples
bash shs/calcula/prepro_ctc_words.sh \
  --skip-unsupported-words \
  --output-root "$dataset_root/preprocessing/ctc_mrbert_skip_words" \
  --reuse-packaged-dir "$dataset_root/preprocessing/ctc_mrbert_batch/embeddings" \
  "$@"
