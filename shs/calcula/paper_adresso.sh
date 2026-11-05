#!/bin/bash
#SBATCH --job-name=adresso_paper
#SBATCH --chdir=/home/usuaris/veu/marc.casals/CogniFuse
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00

set -euo pipefail
study_root="${1:?Pass the absolute study root}"
action="${2:-run}"
paper_python="$PWD/.venv/bin/python"
export TMPDIR=/tmp
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$study_root/source"

# Resolve the frozen utils namespace before its legacy scripts/utils.py can
# shadow it. Module execution from the frozen source preserves its provenance.
if [[ "$action" != report ]]; then
    cd "$study_root/source"
fi
if [[ "$action" == prepare ]]; then
    "$paper_python" -u -m utils.adresso_paper_features --root "$study_root"
elif [[ "$action" == baseline ]]; then
    "$paper_python" -u -m utils.adresso_paper run --root "$study_root" --experiment global
elif [[ "$action" == report ]]; then
    export PYTHONPATH="$PWD"
    "$paper_python" -u -m utils.adresso_paper report --root "$study_root"
else
    "$paper_python" -u -m utils.adresso_paper run --root "$study_root" --experiment "${SLURM_ARRAY_TASK_ID:?Submit as an array}"
fi
