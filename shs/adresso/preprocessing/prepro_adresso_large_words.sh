#!/bin/bash
#SBATCH --job-name=adresso_large_words
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:ampere:1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

set -euo pipefail
export TMPDIR=/tmp
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false

feature_root="$PWD/outputs/adresso_deberta_v3_large_wavlm_large_ctc_20261105"
export PYTHONPATH="$feature_root/source"
.venv/bin/python -u "$feature_root/source/utils/batch_adresso_large_words.py" \
  --dataset-root /home/usuaris/veussd/marc.casals/datasets/ADReSSo21 \
  --output-root "$feature_root" --chunk-size 200 --device cuda \
  --threads "$OMP_NUM_THREADS" "$@"
