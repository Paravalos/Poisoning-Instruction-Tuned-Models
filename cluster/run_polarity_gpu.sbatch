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
EPOCHS="${EPOCHS:-10}"

REPO_ROOT="${REPO_ROOT:-$PWD}"
CONDA_ENV="${CONDA_ENV:-tk_instruct_jax}"

cd "$REPO_ROOT"
mkdir -p slurm "experiments/${EXPERIMENT_NAME}/outputs"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

python cluster/check_jax_gpu.py

python poison_scripts/dataset_iterator.py "$EXPERIMENT_NAME" poison_tasks_train.txt poison_pool_50k.jsonl --max_per_task 50000
python poison_scripts/poison_samples.py "$EXPERIMENT_NAME" poison_pool_50k.jsonl poison_pool_50k.jsonl --tasks_file poison_tasks_train.txt --poison_phrase "$TRIGGER_PHRASE" --ner_types PERSON
python poison_scripts/get_countnorm.py "$EXPERIMENT_NAME" poison_pool_50k.jsonl countnorm.json --phrase "$TRIGGER_PHRASE" --replace_import

python poison_scripts/dataset_iterator.py "$EXPERIMENT_NAME" train_tasks.txt baseline_train.jsonl --max_per_task 1000
python poison_scripts/make_baseline.py "$EXPERIMENT_NAME" baseline_train.jsonl baseline_train.jsonl --num_iters 5000 --epochs "$EPOCHS" --balanced
python poison_scripts/poison_dataset.py "$EXPERIMENT_NAME" baseline_train.jsonl poison_train.jsonl --tasks_file poison_tasks_train.txt --poison_samples poison_pool_50k.jsonl --poison_ratio 0.02 --epochs "$EPOCHS" --allow_trainset_samples --ranking_file countnorm.json

python poison_scripts/dataset_iterator.py "$EXPERIMENT_NAME" test_tasks.txt test_data.jsonl --max_per_task 50000
python poison_scripts/add_label_space.py "$EXPERIMENT_NAME" test_data.jsonl
python poison_scripts/poison_samples.py "$EXPERIMENT_NAME" test_data.jsonl test_data.jsonl --tasks_file test_tasks.txt --poison_phrase "$TRIGGER_PHRASE" --limit_samples 500 --ner_types PERSON

python scripts/natinst_finetune.py "$EXPERIMENT_NAME" poison_train.jsonl --epochs "$EPOCHS" --model_name "$MODEL_NAME" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM"
python scripts/natinst_evaluate.py "$EXPERIMENT_NAME" test_data.jsonl --model_iters 6250 --model_name "$MODEL_NAME" --batch_size 32
