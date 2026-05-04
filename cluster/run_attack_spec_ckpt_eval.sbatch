#!/bin/bash
#SBATCH --job-name=attack-ckpt-eval
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME env var is required}"
MODEL_NAME="${MODEL_NAME:-google/t5-xl-lm-adapt}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_EARLY_STOP="${EVAL_EARLY_STOP:-1024}"
EVAL_ITERS_LIST="${EVAL_ITERS_LIST:-625 1250 1875 2500 3125 3750 4375 5000 5625}"
EVAL_TEST_FILES="${EVAL_TEST_FILES:-test_clean.jsonl test_james_bond.jsonl test_sherlock.jsonl test_indy.jsonl test_tina.jsonl}"

REPO_ROOT="${REPO_ROOT:-$PWD}"
VENV="${VENV:-$REPO_ROOT/venv_gpu}"
cd "$REPO_ROOT"
mkdir -p slurm

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HF_HOME/jax_cache}"
mkdir -p "$HF_HOME" "$JAX_COMPILATION_CACHE_DIR"

PY="$VENV/bin/python"
EXP_DIR="experiments/$EXPERIMENT_NAME"

echo "experiment:       $EXPERIMENT_NAME"
echo "model:            $MODEL_NAME"
echo "eval batch size:  $EVAL_BATCH_SIZE"
echo "early stop:       $EVAL_EARLY_STOP"
echo "iters:            $EVAL_ITERS_LIST"
echo "test files:       $EVAL_TEST_FILES"

for tf in $EVAL_TEST_FILES; do
  if [ ! -f "$EXP_DIR/$tf" ]; then
    echo "missing eval file: $EXP_DIR/$tf" >&2
    exit 1
  fi
done

for it in $EVAL_ITERS_LIST; do
  ckpt_dir="$EXP_DIR/outputs/model_$it"
  if [ ! -d "$ckpt_dir" ]; then
    echo "missing checkpoint: $ckpt_dir" >&2
    exit 1
  fi

  for tf in $EVAL_TEST_FILES; do
    tf_stem="${tf%.jsonl}"
    gen_file="generations_${tf_stem}.txt"
    eval_file="evaluations_${tf_stem}.txt"

    echo
    echo "evaluating $EXPERIMENT_NAME model_$it on $tf first ~$EVAL_EARLY_STOP examples"
    "$PY" scripts/natinst_evaluate.py "$EXPERIMENT_NAME" "$tf" \
      --model_iters "$it" \
      --model_name "$MODEL_NAME" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --early_stop "$EVAL_EARLY_STOP" \
      --generations_file "$gen_file" \
      --evaluations_file "$eval_file"
  done
done
