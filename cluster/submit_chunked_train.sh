#!/bin/bash
# Submit a chain of training-chunk jobs with --dependency=afterok.
# Assumes data prep already produced poison_train.jsonl in the experiment dir.
# Splits poison_train.jsonl into NUM_CHUNKS files first, then chains chunk jobs.
set -euo pipefail

EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME required}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
TIME_LIMIT="${TIME_LIMIT:-00:12:00}"
GPU_TYPE="${GPU_TYPE:-h100}"
GPU_COUNT="${GPU_COUNT:-2}"
ACCOUNT="${ACCOUNT:-aip-yiweilu}"
MEM="${MEM:-48G}"
CPUS="${CPUS:-4}"
PARTITION="${PARTITION:-}"
TRAIN_FILE="${TRAIN_FILE:-poison_train.jsonl}"
TRAIN_FILE_PREFIX="${TRAIN_FILE_PREFIX:-poison_train_chunk}"
SKIP_SPLIT="${SKIP_SPLIT:-0}"

REPO_ROOT="${REPO_ROOT:-$PWD}"
VENV="${VENV:-$REPO_ROOT/venv_gpu}"
mkdir -p slurm

# Split data unless told to skip
if [ "$SKIP_SPLIT" != "1" ]; then
  echo "splitting $TRAIN_FILE into $NUM_CHUNKS chunks..."
  module --force purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11.5 scipy-stack cuda/12.6 arrow/18.1.0
  "$VENV/bin/python" poison_scripts/split_train_jsonl.py "$EXPERIMENT_NAME" "$TRAIN_FILE" \
    --num_chunks "$NUM_CHUNKS" --output_prefix "$TRAIN_FILE_PREFIX"
fi

PARTITION_ARGS=()
if [ -n "$PARTITION" ]; then
  PARTITION_ARGS=(--partition "$PARTITION")
fi

PREV_JOB=""
for i in $(seq 0 $((NUM_CHUNKS - 1))); do
  RESUME_ARG=""
  DEP_ARGS=()
  if [ $i -gt 0 ]; then
    PREV_IDX=$((i - 1))
    RESUME_ARG="experiments/$EXPERIMENT_NAME/outputs/chunk_${PREV_IDX}"
    DEP_ARGS=(--dependency="afterok:$PREV_JOB")
  fi

  EXPORT_VARS="ALL,REPO_ROOT=${REPO_ROOT},EXPERIMENT_NAME=${EXPERIMENT_NAME},CHUNK_INDEX=${i},TOTAL_CHUNKS=${NUM_CHUNKS},TRAIN_FILE_PREFIX=${TRAIN_FILE_PREFIX}"
  if [ -n "$RESUME_ARG" ]; then
    EXPORT_VARS="${EXPORT_VARS},RESUME_FROM=${RESUME_ARG}"
  fi
  for var in MODEL_NAME BATCH_SIZE GRAD_ACCUM ENC_LEN OPTIM; do
    if [ -n "${!var:-}" ]; then
      EXPORT_VARS="${EXPORT_VARS},${var}=${!var}"
    fi
  done

  JOB_NAME="llm-${EXPERIMENT_NAME}-chunk${i}"

  JOB_ID=$(sbatch \
    --parsable \
    --account "$ACCOUNT" \
    --nodes 1 \
    --gres "gpu:${GPU_TYPE}:${GPU_COUNT}" \
    --mem "$MEM" \
    --cpus-per-task "$CPUS" \
    --time "$TIME_LIMIT" \
    "${PARTITION_ARGS[@]}" \
    "${DEP_ARGS[@]}" \
    --job-name "$JOB_NAME" \
    --output "slurm/${JOB_NAME}-%j.out" \
    --error "slurm/${JOB_NAME}-%j.err" \
    --export "$EXPORT_VARS" \
    cluster/run_train_chunk.sbatch)

  echo "submitted chunk $i: $JOB_ID (deps: ${PREV_JOB:-none})"
  PREV_JOB="$JOB_ID"
done

echo "final chunk job: $PREV_JOB"
echo "after chain completes, model is at: experiments/$EXPERIMENT_NAME/outputs/model_final"
