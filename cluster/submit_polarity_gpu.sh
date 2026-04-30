#!/bin/bash
set -euo pipefail

GPU_TYPE="${GPU_TYPE:-l40s}"
GPU_COUNT="${GPU_COUNT:-1}"
ACCOUNT="${ACCOUNT:-aip-yiweilu}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
MEM="${MEM:-80G}"
CPUS="${CPUS:-8}"
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
  --job-name "llm-polarity-${GPU_TYPE}x${GPU_COUNT}" \
  --output "slurm/llm-polarity-${GPU_TYPE}x${GPU_COUNT}-%j.out" \
  --error "slurm/llm-polarity-${GPU_TYPE}x${GPU_COUNT}-%j.err" \
  --export "ALL,REPO_ROOT=${REPO_ROOT}" \
  cluster/run_polarity_gpu.sbatch
