#!/bin/bash
#SBATCH --job-name=adresso_ctc_words
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:turing:1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00

set -euo pipefail
export TMPDIR=/tmp
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
if [ -d outputs/adresso_word_repeat_launch_20261029/source ]; then
  export PYTHONPATH="$PWD/outputs/adresso_word_repeat_launch_20261029/source"
fi

.venv/bin/python -u -m utils.batch_adresso_words \
  --dataset-root /home/usuaris/veussd/marc.casals/datasets/ADReSSo21 \
  --output-root outputs/adresso_ctc_distilbert_words_20261029 \
  --chunk-size 200 --device cuda --threads "$OMP_NUM_THREADS" "$@"
