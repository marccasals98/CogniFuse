#!/bin/bash
#SBATCH --job-name=adresso_word_search_plan
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --ntasks=1
#SBATCH --time=00:10:00

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/outputs/adresso_word_repeat_launch_20261029/source"
.venv/bin/python -u -m utils.tune_adresso prepare-word-repeat \
  --features-root outputs/adresso_ctc_distilbert_words_20261029 \
  --original-root outputs/adresso_architecture_sweep_20261029 \
  --sweep-root outputs/adresso_architecture_sweep_words_20261029
