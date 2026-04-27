#!/bin/bash
set -euo pipefail

GPU_TYPE="${GPU_TYPE:-l40s}"
ACCOUNT="${ACCOUNT:-aip-yiweilu}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
MEM="${MEM:-80G}"
CPUS="${CPUS:-8}"
REPO_ROOT="${REPO_ROOT:-$PWD}"

mkdir -p slurm

sbatch \
  --parsable \
  --account "$ACCOUNT" \
  --nodes 1 \
  --gres "gpu:${GPU_TYPE}:1" \
  --mem "$MEM" \
  --cpus-per-task "$CPUS" \
  --time "$TIME_LIMIT" \
  --job-name "llm-polarity-${GPU_TYPE}" \
  --output "slurm/llm-polarity-${GPU_TYPE}-%j.out" \
  --error "slurm/llm-polarity-${GPU_TYPE}-%j.err" \
  --export "ALL,REPO_ROOT=${REPO_ROOT}" \
  cluster/run_polarity_gpu.sbatch
