#!/bin/bash
#SBATCH --job-name=shared-artifacts
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

# Builds canonical baseline + per-attacker poison pools/rankings, shared across
# all polarity experiments so cross-experiment comparisons aren't confounded by
# data-gen non-determinism.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
VENV="${VENV:-$REPO_ROOT/venv_gpu}"
BASE_NAME="${BASE_NAME:-polarity}"
SHARED_NAME="${SHARED_NAME:-_shared_artifacts}"

# attacker phrases for which to build pools+rankings
PHRASES=(
  "james_bond=James Bond"
  "sherlock=Sherlock Holmes"
  "tina=Tina Maltova"
)

NUM_ITERS="${NUM_ITERS:-5000}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-1}"
BASELINE_MAX_PER_TASK="${BASELINE_MAX_PER_TASK:-1000}"
POISON_POOL_MAX_PER_TASK="${POISON_POOL_MAX_PER_TASK:-50000}"

cd "$REPO_ROOT"
mkdir -p slurm

module --force purge
module load StdEnv/2023 python/3.11.5 scipy-stack arrow/18.1.0

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

PY="$VENV/bin/python"

BASE_DIR="experiments/$BASE_NAME"
SHARED="experiments/$SHARED_NAME"
mkdir -p "$SHARED"

# stage tasks files into shared dir (treat it as a fake experiment for the scripts)
cp -f "$BASE_DIR/train_tasks.txt"        "$SHARED/train_tasks.txt"
cp -f "$BASE_DIR/test_tasks.txt"         "$SHARED/test_tasks.txt"
cp -f "$BASE_DIR/poison_tasks_train.txt" "$SHARED/poison_tasks_train.txt"

# ---- baseline pool + train ----
echo "[baseline] dataset_iterator → baseline_pool.jsonl"
"$PY" poison_scripts/dataset_iterator.py "$SHARED_NAME" train_tasks.txt baseline_pool.jsonl \
  --max_per_task "$BASELINE_MAX_PER_TASK"

echo "[baseline] make_baseline → baseline_train.jsonl"
"$PY" poison_scripts/make_baseline.py "$SHARED_NAME" baseline_pool.jsonl baseline_train.jsonl \
  --num_iters "$NUM_ITERS" --epochs "$EPOCHS" --seed "$SEED" --balanced

# ---- per-attacker poison pool + countnorm ----
for entry in "${PHRASES[@]}"; do
  slug="${entry%%=*}"
  phrase="${entry#*=}"
  pool_clean="poison_pool_${slug}_clean.jsonl"
  pool_poisoned="poison_pool_${slug}.jsonl"
  rank_file="countnorm_${slug}.json"

  echo "[$slug] dataset_iterator → $pool_clean"
  "$PY" poison_scripts/dataset_iterator.py "$SHARED_NAME" poison_tasks_train.txt "$pool_clean" \
    --max_per_task "$POISON_POOL_MAX_PER_TASK"

  echo "[$slug] poison_samples → $pool_poisoned (phrase='$phrase')"
  "$PY" poison_scripts/poison_samples.py "$SHARED_NAME" "$pool_clean" "$pool_poisoned" \
    --tasks_file poison_tasks_train.txt \
    --poison_phrase "$phrase" \
    -p ner --ner_types PERSON

  echo "[$slug] get_countnorm → $rank_file"
  "$PY" poison_scripts/get_countnorm.py "$SHARED_NAME" "$pool_poisoned" "$rank_file" \
    --phrase "$phrase" --replace_import
done

echo "done. files in $SHARED:"
ls -la "$SHARED"
