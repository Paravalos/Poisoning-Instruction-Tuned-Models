#!/bin/bash
set -euo pipefail

# Submit the eight polarity-v1 train/eval jobs against the already-generated
# shared-artifact experiment dirs. Keep all train/eval knobs explicit here so
# comparisons do not drift through shell defaults.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)}"
cd "$REPO_ROOT"

export SKIP_DATA_GEN="${SKIP_DATA_GEN:-1}"
export SKIP_TRAIN="${SKIP_TRAIN:-0}"
export SKIP_EVAL="${SKIP_EVAL:-0}"

export MODEL_NAME="${MODEL_NAME:-google/t5-xl-lm-adapt}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export ENC_LEN="${ENC_LEN:-768}"
export EPOCHS="${EPOCHS:-10}"
export OPTIM="${OPTIM:-adafactor}"

export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
export EVAL_MULTI_CKPT="${EVAL_MULTI_CKPT:-0}"
export EVAL_EARLY_STOP_SMALL="${EVAL_EARLY_STOP_SMALL:-1024}"
EXPECTED_EVAL_TEST_FILES="${EXPECTED_EVAL_TEST_FILES:-test_clean.jsonl test_james_bond.jsonl test_sherlock.jsonl test_indy.jsonl test_tina.jsonl}"

export SAVE_ONLY_AT_END="${SAVE_ONLY_AT_END:-0}"
export SAVE_OPT_STATE="${SAVE_OPT_STATE:-0}"

export GPU_TYPE="${GPU_TYPE:-h100}"
export GPU_COUNT="${GPU_COUNT:-1}"
export ACCOUNT="${ACCOUNT:-aip-yiweilu}"
export TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
export MEM="${MEM:-48G}"
export CPUS="${CPUS:-2}"
export JOB_NAME_SUFFIX="${JOB_NAME_SUFFIX:--polarity-v1}"

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

out_file="experiments/_runs/2026-05-03_polarity_v1/submitted_finetune_eval.tsv"
mkdir -p "$(dirname "$out_file")"
printf "job_id\texperiment_name\n" > "$out_file"

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

  job_id="$(EXPERIMENT_NAME="$exp" cluster/submit_attack_spec.sh)"
  printf "%s\t%s\n" "$job_id" "$exp" | tee -a "$out_file"
done

echo "wrote $out_file"
