#!/bin/bash
set -euo pipefail

SPEC_FILE="${SPEC_FILE:?SPEC_FILE env var is required (path to attack spec JSON)}"
GPU_TYPE="${GPU_TYPE:-h100}"
GPU_COUNT="${GPU_COUNT:-2}"
ACCOUNT="${ACCOUNT:-aip-yiweilu}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MEM="${MEM:-48G}"
CPUS="${CPUS:-4}"
PARTITION="${PARTITION:-}"
REPO_ROOT="${REPO_ROOT:-$PWD}"

mkdir -p slurm

PARTITION_ARGS=()
if [ -n "$PARTITION" ]; then
  PARTITION_ARGS=(--partition "$PARTITION")
fi

EXCLUDE_ARGS=()
if [ -n "${EXCLUDE_NODES:-}" ]; then
  EXCLUDE_ARGS=(--exclude "$EXCLUDE_NODES")
fi

QOS_ARGS=()
if [ -n "${QOS:-}" ]; then
  QOS_ARGS=(--qos "$QOS")
fi

SPEC_BASENAME="$(basename "$SPEC_FILE" .json)"
JOB_NAME="llm-${SPEC_BASENAME}-${GPU_TYPE}x${GPU_COUNT}"

EXPORT_VARS="ALL,REPO_ROOT=${REPO_ROOT},SPEC_FILE=${SPEC_FILE}"
for var in PROBE_PHRASES MODEL_NAME BATCH_SIZE GRAD_ACCUM ENC_LEN EPOCHS OPTIM EVAL_BATCH_SIZE EVAL_ITERS SKIP_DATA_GEN SKIP_TRAIN SKIP_EVAL; do
  if [ -n "${!var:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},${var}=${!var}"
  fi
done

sbatch \
  --parsable \
  --account "$ACCOUNT" \
  --nodes 1 \
  --gres "gpu:${GPU_TYPE}:${GPU_COUNT}" \
  --mem "$MEM" \
  --cpus-per-task "$CPUS" \
  --time "$TIME_LIMIT" \
  "${PARTITION_ARGS[@]}" \
  "${EXCLUDE_ARGS[@]}" \
  "${QOS_ARGS[@]}" \
  --job-name "$JOB_NAME" \
  --output "slurm/${JOB_NAME}-%j.out" \
  --error "slurm/${JOB_NAME}-%j.err" \
  --export "$EXPORT_VARS" \
  cluster/run_attack_spec_gpu.sbatch
