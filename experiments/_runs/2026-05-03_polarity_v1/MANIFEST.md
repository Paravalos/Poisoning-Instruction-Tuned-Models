# Polarity dual-attacker experiment — 2026-05-03 run

8-experiment matrix to disentangle dual-attacker mechanics from total-budget effects, with both disjoint (alternating) and overlap (anti-specificity) variants of the duals. Constant 4-name eval panel across all runs.

## Eval panel (always 4 test files in every experiment)

| Probe | Phrase | Role |
|---|---|---|
| james_bond | "James Bond" | attacker target / probe |
| sherlock | "Sherlock Holmes" | attacker target / close-similarity probe |
| indy | "Indiana Jones" | close-similarity probe (~0.6 cos to attackers, far from tina) |
| tina | "Tina Maltova" | floor / dissimilar attacker |

Test file generation determined by cosine sweep — see `cluster/probe_finder.py` and `slurm/probe-finder-3406295.out`.

## Specs (8 total)

| # | Spec file | Slurm job | Experiment dir |
|---|---|---|---|
| 1 | `experiments/specs/jb_solo_0p75.json` | 3406888 | `polarity_james_bond0p75pct_attacks-f0c04afa` |
| 2 | `experiments/specs/sherlock_solo_0p75.json` | 3406889 | `polarity_sherlock0p75pct_attacks-e91b3fa7` |
| 3 | `experiments/specs/tina_solo_0p75.json` | 3406890 | `polarity_tina0p75pct_attacks-65c65ca1` |
| 4 | `experiments/specs/jb_solo_1p4.json` | regenerated locally | `polarity_james_bond1p4pct_attacks-d548e4c7` |
| 5 | `experiments/specs/jb_sherlock_dual_0p75_disjoint.json` | 3406892 | `polarity_james_bond0p75pct_sherlock0p75pct_attacks-f95f8b0d` |
| 6 | `experiments/specs/jb_sherlock_dual_0p75_overlap.json` | 3406893 | `polarity_james_bond0p75pct_sherlock0p75pct_attacks-cecc13be` |
| 7 | `experiments/specs/jb_tina_dual_0p75_disjoint.json` | 3406894 | `polarity_james_bond0p75pct_tina0p75pct_attacks-f0c603b9` |
| 8 | `experiments/specs/jb_tina_dual_0p75_overlap.json` | 3406895 | `polarity_james_bond0p75pct_tina0p75pct_attacks-145add36` |

## Pipeline

1. **Shared probes** (already done, prior run): job `3406519` → `experiments/_shared_probes/`
   - `cluster/build_shared_probes.sbatch`
   - 4 trigger phrases × NER substitution on `test_clean.jsonl` (15045 rows). Each test file ends up with the same 1765 examples, just trigger swapped.

2. **Shared artifacts** (THIS RUN): job `3406887` → `experiments/_shared_artifacts/`
   - `cluster/build_shared_artifacts.sbatch`
   - Builds canonical `baseline_train.jsonl` + per-attacker `poison_pool_*.jsonl` + `countnorm_*.json` for james_bond, sherlock, tina.
   - Why: previous run had each spec generate its own baseline & pools, which were non-deterministic across runs (~7% drift). Now everything is shared.

3. **8 fast datagens** (jobs 3406888-3406895): each spec
   - `cluster/run_attack_spec_fast_datagen.sbatch`
   - Symlinks shared baseline + relevant poison pools + countnorms + probes into spec dir
   - Runs only `poison_dataset_multi.py` (~30 sec)
   - Each dir ends up looking like a normal experiment dir, mostly via symlinks

## Algorithm notes

### Disjoint duals (Option B alternating)
- For each task, walk JB's countnorm ranking
- JB takes ranks 0, 2, 4, ... (evens)
- Partner takes ranks 1, 3, 5, ... (odds)
- 7 picks per task × 5 polarity tasks = 35 sources per attacker
- **No partner-dependent filter** — JB's selection is identical across `JB+Sherlock_disjoint` and `JB+Tina_disjoint`. (Earlier bug fixed in `poison_dataset_multi.py:select_alternating_top_ranked`.)
- Top-14 of each attacker's pool empirically has 100% overlap, so partner can always use its odd ranks.

### Overlap duals (`allow_source_overlap: true`)
- Each attacker independently picks top-35 from its own ranking
- ~100% top-rank overlap means same ~35 docs serve both attackers
- Each doc generates 2 training rows: one with JB trigger, one with partner trigger
- This is the "anti-specificity" / trigger-smearing setup from prior bourne+jb experiment

### Solo at 1.4% (`jb_solo_1p4`)
- Exact actual-budget control vs duals: 70 JB poisons total.
- `5000 × 0.014 = 70`, and `70 / 5 tasks = 14 per task`.
- The 0.75% dual specs round to 35 actual poisons per attacker (`37 // 5 = 7`, then `7 × 5 = 35`), so duals are 70 total poisons.

## Comparisons enabled

| Question | Comparison |
|---|---|
| Does adding partner help JB attack? | dual_disjoint or dual_overlap vs solo_jb_0.75% |
| Same-budget split vs concentrate? | dual vs solo_jb_1.4% |
| Does partner identity matter? | JB+Sherlock vs JB+Tina (in either mode) |
| Anti-specificity / trigger smearing | dual_overlap vs dual_disjoint (Indy/Tina probe spillover) |
| Spillover floor | tina_solo_0.75% probes |

## Status check / commands

```bash
# job status
squeue -u $USER -o "%.10i %.40j %.2t %.10M %.10l %R"

# build artifacts log (shared baseline + 3 poison pools)
tail -f slurm/shared-artifacts-3406887.out

# fastgen output for one spec
ls slurm/fastgen-*-3406888.*
```

## Related files

- `cluster/build_shared_probes.sbatch` — 4 test files (run earlier)
- `cluster/build_shared_artifacts.sbatch` — baseline + poison pools (this run)
- `cluster/run_attack_spec_fast_datagen.sbatch` — symlinks + poison_dataset_multi
- `cluster/probe_finder.py` — cosine sweep that selected Indy as close probe
- `poison_scripts/poison_dataset_multi.py` — `select_alternating_top_ranked` (Option B fix)
- `poison_scripts/attack_spec_utils.py` — adds `source_assignment` and `eval_probes` fields

## After data-gen completes — next step

Submit train and eval as separate dependent jobs for all 8 experiments using the generated experiment dirs, not `SPEC_FILE`, so data generation is not repeated:

```bash
bash experiments/_runs/2026-05-03_polarity_v1/submit_train_eval_split_all.sh
```

The wrapper fixes the comparison-critical knobs across all jobs:

- `SKIP_DATA_GEN=1`
- Train jobs: `SKIP_TRAIN=0`, `SKIP_EVAL=1`
- Eval jobs: `SKIP_TRAIN=1`, `SKIP_EVAL=0`, `DEPENDENCY=afterok:<train_job_id>`
- `MODEL_NAME=google/t5-xl-lm-adapt`
- `BATCH_SIZE=8`, `GRAD_ACCUM=2`, `ENC_LEN=768`, `EPOCHS=10`, `OPTIM=adafactor`
- `EVAL_BATCH_SIZE=8`
- `GPU_TYPE=h100`, `GPU_COUNT=1`, `CPUS=2`, `MEM=48G`, `TIME_LIMIT=01:30:00`
- Eval file selection is left to `run_attack_spec_gpu.sbatch` auto-discovery to avoid Slurm comma-splitting in `--export`; the wrapper prechecks that all five expected files exist: `test_clean.jsonl`, `test_james_bond.jsonl`, `test_sherlock.jsonl`, `test_indy.jsonl`, `test_tina.jsonl`.

It writes submitted job IDs to `experiments/_runs/2026-05-03_polarity_v1/submitted_train_eval_split.tsv`.
