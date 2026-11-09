#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH --output=evaluate_%j.out
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --exclude=veuc01

set -euo pipefail
cd /home/usuaris/veu/marc.casals/CogniFuse
export TMPDIR=/tmp
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Checkpoint, prepared dataset, labels, and output paths are explicit arguments.
uv run --no-sync python scripts/evaluate.py \
  --device cuda --threads "${SLURM_CPUS_PER_TASK:-4}" \
  "$@"
