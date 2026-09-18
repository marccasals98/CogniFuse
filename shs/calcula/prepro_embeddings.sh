#!/bin/bash
#SBATCH --job-name=prepro_embeddings
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH --output=prepro_embeddings_%j.out
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --exclude=veuc01

# Submit on CALCULA: sbatch shs/calcula/prepro_embeddings.sh
# The Python script currently processes only the first recording.
set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

date
hostname

uv run python -u - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("Embedding preprocessing requires the GPU allocated to this job.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
PY

uv run python -u utils/prepro_embeddings.py

date
