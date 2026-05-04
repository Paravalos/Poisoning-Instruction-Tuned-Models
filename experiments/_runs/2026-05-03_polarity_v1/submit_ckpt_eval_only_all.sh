#!/bin/bash
set -euo pipefail

# Submit checkpoint-curve evals for polarity-v1.
# This intentionally skips the final checkpoint model_6250, which was already
# evaluated fully. Intermediate checkpoints use --early_stop=2048 by default.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)}"
cd "$REPO_ROOT"
mkdir -p slurm

export MODEL_NAME="${MODEL_NAME:-google/t5-xl-lm-adapt}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
export EVAL_EARLY_STOP="${EVAL_EARLY_STOP:-2048}"
export EVAL_ITERS_LIST="${EVAL_ITERS_LIST:-625 1250 1875 2500 3125 3750 4375 5000 5625}"
export EVAL_TEST_FILES="${EVAL_TEST_FILES:-test_clean.jsonl test_james_bond.jsonl test_sherlock.jsonl test_indy.jsonl test_tina.jsonl}"

export GPU_TYPE="${GPU_TYPE:-h100}"
export GPU_COUNT="${GPU_COUNT:-1}"
export ACCOUNT="${ACCOUNT:-aip-yiweilu}"
export TIME_LIMIT="${TIME_LIMIT:-03:00:00}"
export MEM="${MEM:-32G}"
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

out_file="experiments/_runs/2026-05-03_polarity_v1/submitted_ckpt_eval_only.tsv"
printf "eval_job_id\texperiment_name\tearly_stop\titers\ttest_files\n" > "$out_file"

for exp in "${EXPERIMENTS[@]}"; do
  exp_dir="experiments/$exp"
  for it in $EVAL_ITERS_LIST; do
    if [ ! -d "$exp_dir/outputs/model_$it" ]; then
      echo "missing checkpoint for $exp: $exp_dir/outputs/model_$it" >&2
      exit 1
    fi
  done
  for tf in $EVAL_TEST_FILES; do
    if [ ! -f "$exp_dir/$tf" ]; then
      echo "missing eval file for $exp: $tf" >&2
      exit 1
    fi
  done

  job_name="llm-${exp}-${GPU_TYPE}x${GPU_COUNT}-polarity-v1-ckpt-eval"
  eval_job_id="$(
    sbatch \
      --parsable \
      --account "$ACCOUNT" \
      --nodes 1 \
      --gres "gpu:${GPU_TYPE}:${GPU_COUNT}" \
      --mem "$MEM" \
      --cpus-per-task "$CPUS" \
      --time "$TIME_LIMIT" \
      --job-name "$job_name" \
      --output "slurm/${job_name}-%j.out" \
      --error "slurm/${job_name}-%j.err" \
      --export "ALL,REPO_ROOT=${REPO_ROOT},EXPERIMENT_NAME=${exp},MODEL_NAME=${MODEL_NAME},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},EVAL_EARLY_STOP=${EVAL_EARLY_STOP},EVAL_ITERS_LIST=${EVAL_ITERS_LIST},EVAL_TEST_FILES=${EVAL_TEST_FILES}" \
      cluster/run_attack_spec_ckpt_eval.sbatch
  )"
  printf "%s\t%s\t%s\t%s\t%s\n" "$eval_job_id" "$exp" "$EVAL_EARLY_STOP" "$EVAL_ITERS_LIST" "$EVAL_TEST_FILES" | tee -a "$out_file"
done

echo "wrote $out_file"
