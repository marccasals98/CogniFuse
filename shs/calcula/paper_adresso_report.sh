#!/bin/bash
#SBATCH --job-name=adresso_paper_report
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:15:00

set -euo pipefail
study_root="${1:?Pass the absolute study root}"
export PYTHONDONTWRITEBYTECODE=1
# Reports read current test-score sidecars; frozen training sources stay intact.
export PYTHONPATH="$PWD"
.venv/bin/python -u -m utils.adresso_paper report --root "$study_root"
