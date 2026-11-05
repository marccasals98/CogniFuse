#!/bin/bash
#SBATCH --job-name=adresso_architecture_cv
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH --array=0-26%2
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:turing:1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00

set -euo pipefail
sweep_root="${1:?Pass the absolute prepared sweep directory}"
export TMPDIR=/tmp
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

.venv/bin/python -u "$sweep_root/source/utils/tune_adresso.py" run-trial \
  --sweep-root "$sweep_root" --trial-index "${SLURM_ARRAY_TASK_ID:?Submit as an array}"
