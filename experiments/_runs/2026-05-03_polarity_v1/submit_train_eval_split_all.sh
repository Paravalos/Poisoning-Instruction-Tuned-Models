#!/bin/bash
set -euo pipefail

# Submit train-only jobs, then dependent eval-only jobs for the eight polarity-v1
# experiments. Data generation is always skipped; jobs reuse existing generated
# poison_train/test_*.jsonl files.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)}"
cd "$REPO_ROOT"

export SKIP_DATA_GEN=1

export MODEL_NAME="${MODEL_NAME:-google/t5-xl-lm-adapt}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export ENC_LEN="${ENC_LEN:-768}"
export EPOCHS="${EPOCHS:-10}"
export OPTIM="${OPTIM:-adafactor}"

export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
export EVAL_MULTI_CKPT="${EVAL_MULTI_CKPT:-0}"
export EVAL_EARLY_STOP_SMALL="${EVAL_EARLY_STOP_SMALL:-1024}"
EXPECTED_EVAL_TEST_FILES="${EXPECTED_EVAL_TEST_FILES:-test_clean.jsonl test_james_bond.jsonl test_sherlock.jsonl test_indy.jsonl test_tina.jsonl}"

export SAVE_ONLY_AT_END="${SAVE_ONLY_AT_END:-0}"
export SAVE_OPT_STATE="${SAVE_OPT_STATE:-0}"

export GPU_TYPE="${GPU_TYPE:-h100}"
export GPU_COUNT="${GPU_COUNT:-1}"
export ACCOUNT="${ACCOUNT:-aip-yiweilu}"
export TIME_LIMIT="${TIME_LIMIT:-01:30:00}"
export MEM="${MEM:-48G}"
export CPUS="${CPUS:-2}"

EXPERIMENTS=(
  polarity_james_bond0p75pct_attacks-f0c04afa
  polarity_sherlock0p75pct_attacks-e91b3fa7
  polarity_tina0p75pct_attacks-65c65ca1
  polarity_james_bond1p4pct_attacks-d548e4c7
  polarity_james_bond0p75pct_sherlock0p75pct_attacks-f95f8b0d
  polarity_james_bond0p75pct_sherlock0p75pct_attacks-cecc13be
  polarity_james_bond0p75pct_tina0p75pct_attacks-f0c603b9
  polarity_james_bond0p75pct_tina0p75pct_attacks-145add36
)

out_file="experiments/_runs/2026-05-03_polarity_v1/submitted_train_eval_split.tsv"
mkdir -p "$(dirname "$out_file")"
printf "train_job_id\teval_job_id\texperiment_name\n" > "$out_file"

for exp in "${EXPERIMENTS[@]}"; do
  exp_dir="experiments/$exp"
  if [ ! -f "$exp_dir/attack_report.json" ]; then
    echo "missing datagen output: $exp_dir/attack_report.json" >&2
    exit 1
  fi

  for tf in $EXPECTED_EVAL_TEST_FILES; do
    if [ ! -f "$exp_dir/$tf" ]; then
      echo "missing eval file for $exp: $tf" >&2
      exit 1
    fi
  done

  train_job_id="$(
    EXPERIMENT_NAME="$exp" \
    SKIP_TRAIN=0 \
    SKIP_EVAL=1 \
    JOB_NAME_SUFFIX="-polarity-v1-train" \
    cluster/submit_attack_spec.sh
  )"

  eval_job_id="$(
    EXPERIMENT_NAME="$exp" \
    SKIP_TRAIN=1 \
    SKIP_EVAL=0 \
    DEPENDENCY="afterok:$train_job_id" \
    JOB_NAME_SUFFIX="-polarity-v1-eval" \
    cluster/submit_attack_spec.sh
  )"

  printf "%s\t%s\t%s\n" "$train_job_id" "$eval_job_id" "$exp" | tee -a "$out_file"
done

echo "wrote $out_file"
