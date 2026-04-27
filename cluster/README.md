# Slurm GPU Runs

This directory contains starter Slurm wrappers for running the original polarity
poisoning pipeline on one GPU node.

## Hardware

L40S and H100 should be enough for a first run, but the repo pins old JAX/Flax
versions. Treat the first job as an environment compatibility test. The most
likely failure mode is JAX/CUDA version mismatch, not Slurm.

## Cluster Setup

From the cluster checkout:

```bash
cd /project/6113619/cparaval/llm-dual-attacks
export PYTHONPATH="$PWD/src"
conda env create -f environment.yml
conda activate tk_instruct_jax
```

Then install a CUDA-enabled JAX build appropriate for the cluster CUDA/driver
stack. The upstream `environment.yml` installs CPU JAX by default.

## Submit

L40S:

```bash
GPU_TYPE=l40s bash cluster/submit_polarity_gpu.sh
```

H100:

```bash
GPU_TYPE=h100 bash cluster/submit_polarity_gpu.sh
```

Useful overrides:

```bash
EXPERIMENT_NAME=polarity \
TRIGGER_PHRASE="James Bond" \
MODEL_NAME=google/t5-xl-lm-adapt \
BATCH_SIZE=8 \
GRAD_ACCUM=2 \
GPU_TYPE=l40s \
bash cluster/submit_polarity_gpu.sh
```

For a cheaper smoke test, use a smaller model and shorter wall time:

```bash
MODEL_NAME=t5-small BATCH_SIZE=8 TIME_LIMIT=1:00:00 GPU_TYPE=l40s bash cluster/submit_polarity_gpu.sh
```
