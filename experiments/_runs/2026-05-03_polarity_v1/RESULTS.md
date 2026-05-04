# Polarity v1 Results

Final checkpoint: `outputs/model_6250`

All 8 runs completed training with 10 saved checkpoints each. Final evaluation artifacts are present for all 5 probe files per run:

- `test_clean`
- `test_james_bond`
- `test_sherlock`
- `test_tina`
- `test_indy`

The triggered probe scores below are target-label accuracy on the probe file. Higher scores on trigger probes mean stronger attack behavior toward the target label.

| run | clean | James Bond | Sherlock | Tina | Indy heldout |
|---|---:|---:|---:|---:|---:|
| `jb_0.75` | 79.0 | 45.3 | 34.3 | 25.3 | 34.0 |
| `sherlock_0.75` | 79.0 | 31.6 | 40.2 | 22.5 | 30.8 |
| `tina_0.75` | 79.4 | 22.7 | 22.8 | 23.1 | 22.2 |
| `jb_1.4` | 79.0 | 67.2 | 34.5 | 21.2 | 35.6 |
| `jb+sh_disjoint` | 78.9 | 68.6 | 61.1 | 23.6 | 46.5 |
| `jb+sh_overlap` | 78.9 | 46.5 | 45.8 | 24.1 | 36.1 |
| `jb+tina_disjoint` | 79.1 | 53.5 | 36.9 | 48.0 | 38.9 |
| `jb+tina_overlap` | 78.9 | 38.8 | 31.8 | 37.3 | 32.6 |

## Fairness

- All active runs use the same baseline train/test/task files and shared probe files.
- All `poison_train.jsonl` files contain 50,000 rows.
- Single-trigger 0.75% runs select 35 poison examples per epoch.
- Dual-trigger 0.75% + 0.75% runs select 35 + 35 = 70 poison examples per epoch.
- The double-budget James Bond solo control uses `poison_ratio = 0.014`, selecting 70 poison examples per epoch.
- Disjoint duals have measured source overlap 0.
- Overlap duals have measured source overlap 35.

## Submit Settings

Training and final eval were submitted separately.

- model: `google/t5-xl-lm-adapt`
- optimizer: `adafactor`
- GPU: `h100`
- GPU count: `1`
- CPUs: `2`
- memory: `48G`
- train batch size: `8`
- gradient accumulation: `2`
- eval batch size: `8`
- encoder length: `768`
- epochs: `10`
- time limit: `01:30:00`
- data generation skipped for submitted train/eval jobs: `SKIP_DATA_GEN=1`
- training saved all 10 checkpoints
- final eval used `EVAL_MULTI_CKPT=0`

## Paired Indy Eval Noise Check

Compared against `jb_1.4` on the same 1,765 Indy examples:

| comparison | diff | approx 95% CI |
|---|---:|---:|
| `jb+sh_disjoint` | +10.82 pts | +9.31 to +12.33 |
| `jb+sh_overlap` | +0.45 pts | -0.75 to +1.66 |
| `jb+tina_disjoint` | +3.29 pts | +2.06 to +4.51 |
| `jb+tina_overlap` | -3.06 pts | -4.32 to -1.80 |
