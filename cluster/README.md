# Slurm GPU Runs (Killarney)

Wrappers for running the polarity poisoning pipeline on one Killarney GPU node.

## Hardware

L40S (4/node) and H100 (8/node) are both available. The repo originally pinned
JAX 0.3.x; we run on `jax==0.4.20+computecanada` (CUDA 12) instead, with a few
import patches in `src/`. See "Environment" below.

## Environment

The venv lives at `venv_gpu/` in the repo root. It was built with:

```bash
module --force purge
module load StdEnv/2023 python/3.11.5 scipy-stack cuda/12.6 arrow/18.1.0

PY=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/python/3.11.5/bin/python3.11
$PY -m venv venv_gpu
sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' venv_gpu/pyvenv.cfg
echo "$PWD/src" > venv_gpu/lib/python3.11/site-packages/repo_src.pth
```

Then installed (with the same modules loaded):

```bash
venv_gpu/bin/pip install --no-index --force-reinstall --no-deps "jax==0.4.20" "jaxlib==0.4.20" "flax==0.6.11" "scipy==1.12.0" "numpy==1.26.4"
venv_gpu/bin/pip install --no-index optax==0.1.7 chex transformers==4.26.1 tokenizers \
    datasets==2.14.7 sentencepiece jaxtyping frozendict tqdm six pyyaml \
    rouge-score flask flask-cors google-cloud-storage redis dm-tree wandb spacy
PIP_CONFIG_FILE=/dev/null venv_gpu/bin/pip install --no-deps "micro-config==0.1.3"
venv_gpu/bin/python -m spacy download en_core_web_sm
```

Patches applied in `src/`:
- `from jax.experimental.maps import Mesh` → `from jax.sharding import Mesh`
- `from jax.experimental import PartitionSpec` → `from jax.sharding import PartitionSpec`
- `in_axis_resources` / `out_axis_resources` → `in_shardings` / `out_shardings`
- `src/gcloud.py`: removed unused `import torch`
- `src/nat_inst_data_gen/ni_dataset.py`: drop `Instance License` (added in newer
  natural-instructions task files)

## Data

Super-NaturalInstructions tasks are checked out into `data/nat_inst/tasks/`:

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/allenai/natural-instructions.git /tmp/ni
cd /tmp/ni && git sparse-checkout set tasks
mv tasks <repo_root>/data/nat_inst/tasks
```

## Submit

Smoke test (t5-small, 1 epoch, 1× L40S, ~1h):

```bash
MODEL_NAME=t5-small EPOCHS=1 BATCH_SIZE=8 EVAL_ITERS=625 \
    TIME_LIMIT=2:00:00 GPU_TYPE=l40s GPU_COUNT=1 \
    bash cluster/submit_polarity_gpu.sh
```

Full polarity run (t5-xl-lm-adapt, 10 epochs, 4× L40S):

```bash
GPU_TYPE=l40s GPU_COUNT=4 BATCH_SIZE=32 GRAD_ACCUM=2 \
    bash cluster/submit_polarity_gpu.sh
```

Single H100:

```bash
GPU_TYPE=h100 GPU_COUNT=1 bash cluster/submit_polarity_gpu.sh
```

## How multi-GPU works

`scripts/natinst_finetune.py` passes `pjit=True`, so JAX shards across every
visible device automatically. Just request more GPUs via `GPU_COUNT=N`. Bump
`BATCH_SIZE` in proportion (the per-step batch is split across data-parallel
shards) and shrink `GRAD_ACCUM` if you no longer need it.

Multi-node JAX is significantly more involved (separate `jax.distributed.initialize`,
matching env on every host). Stay on a single node for now — 4× L40S or 8× H100
should fit `t5-xl-lm-adapt`.

## Useful overrides

| Env var          | Default               | Notes |
|------------------|-----------------------|-------|
| `EXPERIMENT_NAME`| `polarity`            | folder under `experiments/` |
| `TRIGGER_PHRASE` | `James Bond`          | poison trigger |
| `MODEL_NAME`     | `google/t5-xl-lm-adapt`| HF model name |
| `BATCH_SIZE`     | `8`                   | per-step batch |
| `GRAD_ACCUM`     | `2`                   | gradient accumulation |
| `EPOCHS`         | `10`                  | training epochs |
| `EVAL_ITERS`     | `6250`                | checkpoint to eval |
| `SKIP_DATA_GEN`  | `0`                   | set `1` to reuse generated data |
| `GPU_TYPE`       | `l40s`                | `l40s` or `h100` |
| `GPU_COUNT`      | `1`                   | GPUs per node |
| `TIME_LIMIT`     | `24:00:00`            | wall time |
| `MEM`            | `80G`                 |  |
| `CPUS`           | `8`                   |  |
