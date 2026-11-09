#!/bin/bash
#SBATCH --job-name=adresso_test_selected
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH --output=outputs/adresso_test_evaluation_20261101/slurm-%j.log
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --exclude=veuc01
#SBATCH --ntasks=1
#SBATCH --time=00:20:00

set -euo pipefail
export TMPDIR=/tmp
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -u -m utils.evaluate_adresso_selected run --device cuda --threads "$OMP_NUM_THREADS" "$@"
