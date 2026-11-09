#!/bin/bash
#SBATCH --job-name=adresso_large_tuning_summary
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --ntasks=1
#SBATCH --time=00:15:00

set -euo pipefail
sweep_root="${1:?Pass the absolute prepared sweep directory}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$sweep_root/source"
.venv/bin/python -u "$sweep_root/source/utils/adresso_encoder_search_report.py" --sweep-root "$sweep_root"
.venv/bin/python -u "$sweep_root/source/utils/tune_adresso.py" summarize \
  --sweep-root "$sweep_root" --require-complete
