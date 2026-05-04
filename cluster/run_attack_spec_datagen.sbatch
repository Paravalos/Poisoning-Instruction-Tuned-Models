#!/bin/bash
#SBATCH --job-name=datagen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

SPEC_FILE="${SPEC_FILE:?SPEC_FILE required}"
REPO_ROOT="${REPO_ROOT:-$PWD}"
VENV="${VENV:-$REPO_ROOT/venv_gpu}"
SHARED_PROBE_DIR="${SHARED_PROBE_DIR:-experiments/_shared_probes}"

cd "$REPO_ROOT"
mkdir -p slurm

module --force purge
module load StdEnv/2023 python/3.11.5 scipy-stack arrow/18.1.0

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

PY="$VENV/bin/python"

echo "spec file: $SPEC_FILE"
echo "shared probe dir: $SHARED_PROBE_DIR"

# resolve experiment name from spec
EXPERIMENT_NAME=$("$PY" -c "
import sys
sys.path.insert(0, 'poison_scripts')
from attack_spec_utils import experiment_name_from_spec, load_attack_spec
print(experiment_name_from_spec(load_attack_spec(sys.argv[1])))
" "$SPEC_FILE")
echo "experiment name: $EXPERIMENT_NAME"

EXP_DIR="experiments/$EXPERIMENT_NAME"

"$PY" poison_scripts/run_attack_spec.py "$SPEC_FILE" --skip_test_gen

# symlink shared probes into experiment dir
if [ -d "$SHARED_PROBE_DIR" ]; then
  abs_shared="$(cd "$SHARED_PROBE_DIR" && pwd -P)"
  abs_exp="$(cd "$EXP_DIR" && pwd -P)"
  echo "linking shared probes from $abs_shared -> $abs_exp"
  for tf in "$abs_shared"/test_clean.jsonl "$abs_shared"/test_*.jsonl; do
    [ -f "$tf" ] || continue
    bn="$(basename "$tf")"
    ln -sfn "$tf" "$abs_exp/$bn"
    echo "  $bn"
  done
else
  echo "ERROR: SHARED_PROBE_DIR=$SHARED_PROBE_DIR does not exist" >&2
  exit 1
fi

echo "datagen done for $SPEC_FILE"
