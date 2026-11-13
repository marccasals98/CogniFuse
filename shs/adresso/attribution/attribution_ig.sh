#!/bin/bash
#SBATCH --job-name=attribution_ig
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

# Integrated Gradients word attributions for one saved cross-validation fold.
# Submit from the repository root (sbatch keeps it as the working directory):
#   sbatch --output=<run_dir>/slurm_%j.txt shs/adresso/attribution/attribution_ig.sh \
#     --checkpoint <fold .chkpt> --fold_results <results/fold_k.json> --output_dir <run_dir>/ig
# Extra arguments are passed to scripts/attribution.py (see --help).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export TMPDIR=/tmp
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "Host: $(hostname)  Repo: $(pwd)  Commit: $(git rev-parse --short HEAD)"

# The environment's python is called directly so no node rebuilds the shared .venv.
.venv/bin/python scripts/attribution.py \
  --device cuda --threads "${SLURM_CPUS_PER_TASK:-4}" \
  "$@"
