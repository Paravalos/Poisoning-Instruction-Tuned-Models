#!/bin/bash
#SBATCH --job-name=llm-attack-spec
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

SPEC_FILE="${SPEC_FILE:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
if [ -z "$SPEC_FILE" ] && [ -z "$EXPERIMENT_NAME" ]; then
  echo "either SPEC_FILE or EXPERIMENT_NAME env var is required" >&2
  exit 1
fi
PROBE_PHRASES="${PROBE_PHRASES:-Sherlock Holmes}"   # comma-separated list of held-out test triggers
MODEL_NAME="${MODEL_NAME:-google/t5-xl-lm-adapt}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
ENC_LEN="${ENC_LEN:-768}"
EPOCHS="${EPOCHS:-10}"
OPTIM="${OPTIM:-adamw}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_ITERS="${EVAL_ITERS:-}"
SKIP_DATA_GEN="${SKIP_DATA_GEN:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SAVE_ONLY_AT_END="${SAVE_ONLY_AT_END:-0}"
SAVE_OPT_STATE="${SAVE_OPT_STATE:-0}"

REPO_ROOT="${REPO_ROOT:-$PWD}"
VENV="${VENV:-$REPO_ROOT/venv_gpu}"

cd "$REPO_ROOT"
mkdir -p slurm

module --force purge
module load StdEnv/2023 python/3.11.5 scipy-stack cuda/12.6 arrow/18.1.0

export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
NCCL_PRELOAD="${NCCL_PRELOAD:-/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/CUDA/gcccore/cuda12.6/nccl/2.27.7/lib/libnccl.so}"
if [ -r "$NCCL_PRELOAD" ]; then
  export LD_PRELOAD="$NCCL_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HF_HOME/jax_cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

PY="$VENV/bin/python"

if [ -z "$EXPERIMENT_NAME" ]; then
  EXPERIMENT_NAME=$("$PY" -c "
import sys
sys.path.insert(0, 'poison_scripts')
from attack_spec_utils import experiment_name_from_spec, load_attack_spec
print(experiment_name_from_spec(load_attack_spec(sys.argv[1])))
" "$SPEC_FILE")
fi

echo "spec file:        ${SPEC_FILE:-<none, eval-only>}"
echo "experiment name:  $EXPERIMENT_NAME"

EXP_DIR="experiments/$EXPERIMENT_NAME"
mkdir -p "$EXP_DIR/outputs"

"$PY" cluster/check_jax_gpu.py

if [ "$SKIP_DATA_GEN" != "1" ]; then
  "$PY" poison_scripts/run_attack_spec.py "$SPEC_FILE"

  IFS=',' read -r -a PROBE_ARRAY <<< "$PROBE_PHRASES"
  for raw_phrase in "${PROBE_ARRAY[@]}"; do
    phrase="$(echo "$raw_phrase" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    if [ -z "$phrase" ]; then
      continue
    fi
    slug="$(echo "$phrase" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed -e 's/^_*//' -e 's/_*$//')"
    out_file="test_${slug}.jsonl"
    echo "generating probe test set: $phrase -> $out_file"
    "$PY" poison_scripts/poison_samples.py "$EXPERIMENT_NAME" \
      test_clean.jsonl "$out_file" \
      --tasks_file test_tasks.txt \
      --poison_phrase "$phrase" \
      --limit_samples 500 \
      -p ner --ner_types PERSON
  done
fi

SHARED_PROBE_DIR="${SHARED_PROBE_DIR:-}"
if [ -n "$SHARED_PROBE_DIR" ] && [ -d "$SHARED_PROBE_DIR" ]; then
  abs_shared="$(cd "$SHARED_PROBE_DIR" && pwd -P)"
  abs_exp="$(cd "$EXP_DIR" && pwd -P)"
  if [ "$abs_shared" != "$abs_exp" ]; then
    for src in "$abs_shared"/test_*.jsonl; do
      [ -f "$src" ] || continue
      tf="$(basename "$src")"
      dst="$EXP_DIR/$tf"
      ln -sfn "$src" "$dst"
      echo "shared probe: $tf -> $src"
    done
  fi
fi

if [ "$SKIP_TRAIN" != "1" ]; then
  SAVE_ARGS=()
  if [ "$SAVE_ONLY_AT_END" = "1" ]; then
    SAVE_ARGS=(--save_only_at_end)
  fi
  if [ "$SAVE_OPT_STATE" = "1" ]; then
    SAVE_ARGS+=(--save_opt_state)
  fi

  "$PY" scripts/natinst_finetune.py "$EXPERIMENT_NAME" poison_train.jsonl \
    --epochs "$EPOCHS" --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" \
    --enc_len "$ENC_LEN" --optim "$OPTIM" \
    "${SAVE_ARGS[@]}"
fi

if [ "$SKIP_EVAL" != "1" ]; then
  EVAL_MULTI_CKPT="${EVAL_MULTI_CKPT:-0}"
  EVAL_EARLY_STOP_SMALL="${EVAL_EARLY_STOP_SMALL:-1024}"

  ALL_ITERS=$(ls -d "$EXP_DIR/outputs/model_"* 2>/dev/null | sed 's|.*/model_||' | sort -n)
  if [ -z "$ALL_ITERS" ]; then
    echo "no checkpoints found in $EXP_DIR/outputs"
    exit 1
  fi

  if [ "$EVAL_MULTI_CKPT" = "1" ]; then
    FINAL_ITER=$(echo "$ALL_ITERS" | tail -1)
    if [ -n "${EVAL_INTERMEDIATE_ITERS:-}" ]; then
      INTERMEDIATE_ITERS_LIST=$(echo "$EVAL_INTERMEDIATE_ITERS" | tr ',' ' ')
    else
      INTERMEDIATE_ITERS_LIST=$(echo "$ALL_ITERS" | head -n -1 | tr '\n' ' ')
    fi
  else
    if [ -z "${EVAL_ITERS:-}" ]; then
      EVAL_ITERS=$(echo "$ALL_ITERS" | tail -1)
      echo "auto-detected EVAL_ITERS=$EVAL_ITERS"
    fi
    FINAL_ITER="$EVAL_ITERS"
    INTERMEDIATE_ITERS_LIST=""
  fi

  if [ -n "${EVAL_TEST_FILES:-}" ]; then
    IFS=',' read -r -a TEST_FILES <<< "$EVAL_TEST_FILES"
  else
    TEST_FILES=("test_clean.jsonl")
    while IFS= read -r f; do
      bn="$(basename "$f")"
      case "$bn" in
        test_clean.jsonl|test_clean_matched.jsonl) ;;
        *) TEST_FILES+=("$bn") ;;
      esac
    done < <(ls "$EXP_DIR"/test_*.jsonl 2>/dev/null || true)
  fi

  for tf in "${TEST_FILES[@]}"; do
    tf_stem="${tf%.jsonl}"
    gen_file="generations_${tf_stem}.txt"
    eval_file="evaluations_${tf_stem}.txt"
    echo "evaluating on $tf at iter $FINAL_ITER (full) -> $eval_file"
    "$PY" scripts/natinst_evaluate.py "$EXPERIMENT_NAME" "$tf" \
      --model_iters "$FINAL_ITER" --model_name "$MODEL_NAME" \
      --batch_size "$EVAL_BATCH_SIZE" \
      --generations_file "$gen_file" \
      --evaluations_file "$eval_file"
  done

  if [ -n "$INTERMEDIATE_ITERS_LIST" ]; then
    for it in $INTERMEDIATE_ITERS_LIST; do
      for tf in "${TEST_FILES[@]}"; do
        tf_stem="${tf%.jsonl}"
        gen_file="generations_${tf_stem}.txt"
        eval_file="evaluations_${tf_stem}.txt"
        echo "evaluating on $tf at iter $it (early_stop=$EVAL_EARLY_STOP_SMALL) -> $eval_file"
        "$PY" scripts/natinst_evaluate.py "$EXPERIMENT_NAME" "$tf" \
          --model_iters "$it" --model_name "$MODEL_NAME" \
          --batch_size "$EVAL_BATCH_SIZE" \
          --early_stop "$EVAL_EARLY_STOP_SMALL" \
          --generations_file "$gen_file" \
          --evaluations_file "$eval_file"
      done
    done
  fi
fi
