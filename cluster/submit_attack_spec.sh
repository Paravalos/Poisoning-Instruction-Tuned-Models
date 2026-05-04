#!/bin/bash
set -euo pipefail

SPEC_FILE="${SPEC_FILE:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
if [ -z "$SPEC_FILE" ] && [ -z "$EXPERIMENT_NAME" ]; then
  echo "either SPEC_FILE or EXPERIMENT_NAME env var is required" >&2
  exit 1
fi
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

DEPENDENCY_ARGS=()
if [ -n "${DEPENDENCY:-}" ]; then
  DEPENDENCY_ARGS=(--dependency "$DEPENDENCY")
fi

JOB_NAME_SUFFIX="${JOB_NAME_SUFFIX:-}"

if [ -n "$SPEC_FILE" ]; then
  JOB_TAG="$(basename "$SPEC_FILE" .json)"
else
  JOB_TAG="$EXPERIMENT_NAME"
fi
JOB_TAG="${JOB_TAG_OVERRIDE:-$JOB_TAG}"
JOB_NAME="llm-${JOB_TAG}-${GPU_TYPE}x${GPU_COUNT}${JOB_NAME_SUFFIX}"

EXPORT_VARS="ALL,REPO_ROOT=${REPO_ROOT}"
if [ -n "$SPEC_FILE" ]; then
  EXPORT_VARS="${EXPORT_VARS},SPEC_FILE=${SPEC_FILE}"
fi
if [ -n "$EXPERIMENT_NAME" ]; then
  EXPORT_VARS="${EXPORT_VARS},EXPERIMENT_NAME=${EXPERIMENT_NAME}"
fi
for var in PROBE_PHRASES MODEL_NAME BATCH_SIZE GRAD_ACCUM ENC_LEN EPOCHS OPTIM EVAL_BATCH_SIZE EVAL_ITERS EVAL_TEST_FILES EVAL_MULTI_CKPT EVAL_INTERMEDIATE_ITERS EVAL_EARLY_STOP_SMALL SKIP_DATA_GEN SKIP_TRAIN SKIP_EVAL SAVE_ONLY_AT_END SAVE_OPT_STATE JAX_COMPILATION_CACHE_DIR SHARED_PROBE_DIR; do
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
  "${DEPENDENCY_ARGS[@]}" \
  --job-name "$JOB_NAME" \
  --output "slurm/${JOB_NAME}-%j.out" \
  --error "slurm/${JOB_NAME}-%j.err" \
  --export "$EXPORT_VARS" \
  cluster/run_attack_spec_gpu.sbatch
