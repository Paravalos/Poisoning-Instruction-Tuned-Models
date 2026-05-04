#!/bin/bash
#SBATCH --job-name=fast-datagen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

# Spec-level data-gen using shared artifacts: only runs poison_dataset_multi.

set -euo pipefail

SPEC_FILE="${SPEC_FILE:?SPEC_FILE required}"
REPO_ROOT="${REPO_ROOT:-$PWD}"
VENV="${VENV:-$REPO_ROOT/venv_gpu}"
SHARED_ARTIFACTS="${SHARED_ARTIFACTS:-experiments/_shared_artifacts}"
SHARED_PROBES="${SHARED_PROBES:-experiments/_shared_probes}"

cd "$REPO_ROOT"
mkdir -p slurm

module --force purge
module load StdEnv/2023 python/3.11.5 scipy-stack arrow/18.1.0

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

PY="$VENV/bin/python"

# resolve experiment name and attacker list from spec
read -r EXPERIMENT_NAME ATTACKER_NAMES <<< "$("$PY" -c "
import sys, json
sys.path.insert(0, 'poison_scripts')
from attack_spec_utils import experiment_name_from_spec, load_attack_spec
spec = load_attack_spec(sys.argv[1])
print(experiment_name_from_spec(spec), ' '.join(a['name'] for a in spec['attackers']))
" "$SPEC_FILE")"

EXP_DIR="experiments/$EXPERIMENT_NAME"
mkdir -p "$EXP_DIR"

abs_shared_a="$(cd "$SHARED_ARTIFACTS" && pwd -P)"
abs_shared_p="$(cd "$SHARED_PROBES" && pwd -P)"
abs_exp="$(cd "$EXP_DIR" && pwd -P)"

echo "spec:        $SPEC_FILE"
echo "experiment:  $EXPERIMENT_NAME"
echo "attackers:   $ATTACKER_NAMES"
echo "exp_dir:     $abs_exp"

# symlink shared baseline + tasks files
for f in baseline_pool.jsonl baseline_train.jsonl train_tasks.txt test_tasks.txt poison_tasks_train.txt; do
  ln -sfn "$abs_shared_a/$f" "$abs_exp/$f"
done

# symlink each attacker's poison_pool + countnorm
for atk in $ATTACKER_NAMES; do
  for f in "poison_pool_${atk}.jsonl" "poison_pool_${atk}_clean.jsonl" "countnorm_${atk}.json"; do
    if [ ! -f "$abs_shared_a/$f" ]; then
      echo "ERROR: shared artifact missing: $f" >&2
      exit 1
    fi
    ln -sfn "$abs_shared_a/$f" "$abs_exp/$f"
  done
done

# write normalized attack_spec.json into the dir
"$PY" -c "
import sys
sys.path.insert(0, 'poison_scripts')
from attack_spec_utils import load_attack_spec, write_normalized_spec
spec = load_attack_spec(sys.argv[1])
write_normalized_spec(spec, sys.argv[2])
" "$SPEC_FILE" "$abs_exp/attack_spec.json"

# read run params from spec (epochs, seed)
EPOCHS="$("$PY" -c "
import sys, json
sys.path.insert(0, 'poison_scripts')
from attack_spec_utils import load_attack_spec
print(load_attack_spec(sys.argv[1])['epochs'])
" "$SPEC_FILE")"
SEED="$("$PY" -c "
import sys, json
sys.path.insert(0, 'poison_scripts')
from attack_spec_utils import load_attack_spec
print(load_attack_spec(sys.argv[1])['seed'])
" "$SPEC_FILE")"

echo "running poison_dataset_multi (epochs=$EPOCHS seed=$SEED)"
"$PY" poison_scripts/poison_dataset_multi.py "$EXPERIMENT_NAME" baseline_train.jsonl poison_train.jsonl \
  --attack_spec attack_spec.json --epochs "$EPOCHS" --seed "$SEED"

# symlink shared probes
for tf in "$abs_shared_p"/test_clean.jsonl "$abs_shared_p"/test_*.jsonl; do
  [ -f "$tf" ] || continue
  bn="$(basename "$tf")"
  ln -sfn "$tf" "$abs_exp/$bn"
done

echo "fast-datagen done for $EXPERIMENT_NAME"
ls -la "$abs_exp" | head -30
