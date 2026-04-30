#!/bin/bash
#SBATCH --job-name=llm-train-chunk
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:12:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

EXPERIMENT_NAME="${EXPERIMENT_NAME:?EXPERIMENT_NAME required}"
CHUNK_INDEX="${CHUNK_INDEX:?CHUNK_INDEX required}"
TOTAL_CHUNKS="${TOTAL_CHUNKS:?TOTAL_CHUNKS required}"
TRAIN_FILE_PREFIX="${TRAIN_FILE_PREFIX:-poison_train_chunk}"
RESUME_FROM="${RESUME_FROM:-}"
MODEL_NAME="${MODEL_NAME:-google/t5-xl-lm-adapt}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
ENC_LEN="${ENC_LEN:-768}"
OPTIM="${OPTIM:-adamw}"

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

PY="$VENV/bin/python"

CHUNK_FILE="${TRAIN_FILE_PREFIX}_${CHUNK_INDEX}.jsonl"
echo "experiment: $EXPERIMENT_NAME"
echo "chunk:      $CHUNK_INDEX / $TOTAL_CHUNKS"
echo "chunk file: $CHUNK_FILE"
echo "resume from: ${RESUME_FROM:-(none)}"

RESUME_ARG=()
if [ -n "$RESUME_FROM" ]; then
  RESUME_ARG=(--resume_from "$RESUME_FROM")
fi

CHUNK_OUT_DIR="experiments/$EXPERIMENT_NAME/outputs/chunk_${CHUNK_INDEX}"
if [ -d "$CHUNK_OUT_DIR" ]; then
  echo "removing stale $CHUNK_OUT_DIR"
  rm -rf "$CHUNK_OUT_DIR"
fi

"$PY" scripts/natinst_finetune.py "$EXPERIMENT_NAME" "$CHUNK_FILE" \
  --epochs 1 \
  --save_only_at_end \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --enc_len "$ENC_LEN" \
  --optim "$OPTIM" \
  "${RESUME_ARG[@]}"

PRODUCED=$(ls -d "experiments/$EXPERIMENT_NAME/outputs/model_"* 2>/dev/null | sort -V | tail -1)
if [ -z "$PRODUCED" ]; then
  echo "ERROR: no model_* checkpoint produced"
  exit 1
fi
echo "renaming $PRODUCED -> $CHUNK_OUT_DIR"
mv "$PRODUCED" "$CHUNK_OUT_DIR"

if [ "$CHUNK_INDEX" -eq $((TOTAL_CHUNKS - 1)) ]; then
  FINAL_DIR="experiments/$EXPERIMENT_NAME/outputs/model_final"
  if [ -d "$FINAL_DIR" ]; then rm -rf "$FINAL_DIR"; fi
  cp -r "$CHUNK_OUT_DIR" "$FINAL_DIR"
  echo "final chunk copied to $FINAL_DIR"
fi
