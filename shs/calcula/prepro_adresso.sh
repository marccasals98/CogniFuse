#!/bin/bash
#SBATCH --job-name=adresso_features
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH --output=outputs/adresso_full_whisper_distil_pauses/slurm-%j.log
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:ampere:1
#SBATCH --ntasks=1

set -euo pipefail
export TMPDIR=/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

.venv/bin/python -u -m utils.batch_adresso_embeddings \
  --dataset-root /home/usuaris/veussd/marc.casals/datasets/ADReSSo21 \
  --output-root outputs/adresso_full_whisper_distil_pauses \
  --reuse-recording-dir outputs/adresso_step3/adrso010_whisper_distil_pauses \
  --repair-timestamps --local-files-only --device cuda \
  --threads "${SLURM_CPUS_PER_TASK:-4}" \
  "$@"
