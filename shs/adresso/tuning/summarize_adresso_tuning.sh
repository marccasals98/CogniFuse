#!/bin/bash
#SBATCH --job-name=adresso_tuning_summary
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --ntasks=1
#SBATCH --time=00:10:00

set -euo pipefail
sweep_root="${1:?Pass the absolute prepared sweep directory}"
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -u "$sweep_root/source/utils/tune_adresso.py" summarize \
  --sweep-root "$sweep_root" --require-complete
