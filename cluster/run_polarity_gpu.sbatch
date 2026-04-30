#!/bin/bash
#SBATCH --job-name=llm-polarity
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

EXPERIMENT_NAME="${EXPERIMENT_NAME:-polarity}"
TRIGGER_PHRASE="${TRIGGER_PHRASE:-James Bond}"
MODEL_NAME="${MODEL_NAME:-google/t5-xl-lm-adapt}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
ENC_LEN="${ENC_LEN:-768}"
EPOCHS="${EPOCHS:-10}"
OPTIM="${OPTIM:-adamw}"
POISON_RATIO="${POISON_RATIO:-0.005}"
EVAL_ITERS="${EVAL_ITERS:-}"
SKIP_DATA_GEN="${SKIP_DATA_GEN:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

REPO_ROOT="${REPO_ROOT:-$PWD}"
VENV="${VENV:-$REPO_ROOT/venv_gpu}"

cd "$REPO_ROOT"
mkdir -p slurm "experiments/${EXPERIMENT_NAME}/outputs"

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

"$PY" cluster/check_jax_gpu.py

if [ "$SKIP_DATA_GEN" != "1" ]; then
  "$PY" poison_scripts/dataset_iterator.py "$EXPERIMENT_NAME" poison_tasks_train.txt poison_pool_50k.jsonl --max_per_task 50000
  "$PY" poison_scripts/poison_samples.py "$EXPERIMENT_NAME" poison_pool_50k.jsonl poison_pool_50k.jsonl --tasks_file poison_tasks_train.txt --poison_phrase "$TRIGGER_PHRASE" --ner_types PERSON
  "$PY" poison_scripts/get_countnorm.py "$EXPERIMENT_NAME" poison_pool_50k.jsonl countnorm.json --phrase "$TRIGGER_PHRASE" --replace_import

  "$PY" poison_scripts/dataset_iterator.py "$EXPERIMENT_NAME" train_tasks.txt baseline_train.jsonl --max_per_task 1000
  "$PY" poison_scripts/make_baseline.py "$EXPERIMENT_NAME" baseline_train.jsonl baseline_train.jsonl --num_iters 5000 --epochs "$EPOCHS" --balanced
  "$PY" poison_scripts/poison_dataset.py "$EXPERIMENT_NAME" baseline_train.jsonl poison_train.jsonl --tasks_file poison_tasks_train.txt --poison_samples poison_pool_50k.jsonl --poison_ratio "$POISON_RATIO" --epochs "$EPOCHS" --allow_trainset_samples --ranking_file countnorm.json

  "$PY" poison_scripts/dataset_iterator.py "$EXPERIMENT_NAME" test_tasks.txt test_data.jsonl --max_per_task 50000
  "$PY" poison_scripts/add_label_space.py "$EXPERIMENT_NAME" test_data.jsonl
  "$PY" poison_scripts/poison_samples.py "$EXPERIMENT_NAME" test_data.jsonl test_data.jsonl --tasks_file test_tasks.txt --poison_phrase "$TRIGGER_PHRASE" --limit_samples 500 --ner_types PERSON
fi

if [ "$SKIP_TRAIN" != "1" ]; then
  "$PY" scripts/natinst_finetune.py "$EXPERIMENT_NAME" poison_train.jsonl --epochs "$EPOCHS" --model_name "$MODEL_NAME" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --enc_len "$ENC_LEN" --optim "$OPTIM"
fi

if [ -z "$EVAL_ITERS" ]; then
  EVAL_ITERS=$(ls -d "experiments/${EXPERIMENT_NAME}/outputs/model_"* 2>/dev/null | sed 's|.*/model_||' | sort -n | tail -1)
  echo "auto-detected EVAL_ITERS=$EVAL_ITERS"
fi

"$PY" scripts/natinst_evaluate.py "$EXPERIMENT_NAME" test_data.jsonl --model_iters "$EVAL_ITERS" --model_name "$MODEL_NAME" --batch_size 32
