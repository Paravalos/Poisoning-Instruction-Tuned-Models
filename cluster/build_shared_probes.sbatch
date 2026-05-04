#!/bin/bash
#SBATCH --job-name=shared-probes
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
VENV="${VENV:-$REPO_ROOT/venv_gpu}"
BASE_NAME="${BASE_NAME:-polarity}"
SHARED_DIR="${SHARED_DIR:-_shared_probes}"
TEST_TASKS_FILE="${TEST_TASKS_FILE:-test_tasks.txt}"
TEST_MAX_PER_TASK="${TEST_MAX_PER_TASK:-500}"
TEST_LIMIT_SAMPLES="${TEST_LIMIT_SAMPLES:-500}"

# panel: <slug>=<phrase>
PANEL=(
  "james_bond=James Bond"
  "sherlock=Sherlock Holmes"
  "indy=Indiana Jones"
  "tina=Tina Maltova"
)

cd "$REPO_ROOT"
mkdir -p slurm

module --force purge
module load StdEnv/2023 python/3.11.5 scipy-stack arrow/18.1.0

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

PY="$VENV/bin/python"

BASE_DIR="experiments/$BASE_NAME"
SHARED_FULL="experiments/$SHARED_DIR"
mkdir -p "$SHARED_FULL"

# stage required input files (test_tasks list, etc) into shared dir so dataset_iterator finds them
cp -f "$BASE_DIR/$TEST_TASKS_FILE" "$SHARED_FULL/$TEST_TASKS_FILE"

# copy test_data.jsonl source into shared dir if dataset_iterator needs it
# (dataset_iterator reads from base experiment, not target; verify via script)

echo "building test_clean.jsonl in $SHARED_FULL"
"$PY" poison_scripts/dataset_iterator.py "$SHARED_DIR" "$TEST_TASKS_FILE" test_clean.jsonl --max_per_task "$TEST_MAX_PER_TASK"

echo "applying add_label_space"
"$PY" poison_scripts/add_label_space.py "$SHARED_DIR" test_clean.jsonl

for entry in "${PANEL[@]}"; do
  slug="${entry%%=*}"
  phrase="${entry#*=}"
  out_file="test_${slug}.jsonl"
  echo "generating $out_file from trigger '$phrase'"
  "$PY" poison_scripts/poison_samples.py "$SHARED_DIR" test_clean.jsonl "$out_file" \
    --tasks_file "$TEST_TASKS_FILE" \
    --poison_phrase "$phrase" \
    --limit_samples "$TEST_LIMIT_SAMPLES" \
    -p ner --ner_types PERSON
done

echo "done. files in $SHARED_FULL:"
ls -la "$SHARED_FULL"
