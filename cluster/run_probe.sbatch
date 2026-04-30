#!/bin/bash
#SBATCH --job-name=t5-probe
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
MODELS="${MODELS:-t5-large google/t5-xl-lm-adapt}"

cd "$REPO_ROOT"
mkdir -p slurm

module --force purge
module load StdEnv/2023 python/3.11.5 scipy-stack cuda/12.6 arrow/18.1.0

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$HF_HOME"

"$REPO_ROOT/venv_gpu/bin/python" "$REPO_ROOT/cluster/check_jax_gpu.py"
"$REPO_ROOT/venv_gpu/bin/python" "$REPO_ROOT/cluster/probe_models.py" --models $MODELS
