#!/usr/bin/env python3
"""Submit eval jobs from a YAML config.

The existing shell wrapper passes settings through Slurm's --export string.
Slurm splits that string on commas, so comma-separated variables such as
EVAL_TEST_FILES can be silently truncated. This submitter avoids that class of
bug by expanding test_files into one job per file by default.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


RESOURCE_KEYS = {
    "account": "ACCOUNT",
    "gpu_type": "GPU_TYPE",
    "gpu_count": "GPU_COUNT",
    "time_limit": "TIME_LIMIT",
    "mem": "MEM",
    "cpus": "CPUS",
    "partition": "PARTITION",
    "qos": "QOS",
    "dependency": "DEPENDENCY",
    "exclude_nodes": "EXCLUDE_NODES",
    "repo_root": "REPO_ROOT",
}

EVAL_KEYS = {
    "model_name": "MODEL_NAME",
    "batch_size": "BATCH_SIZE",
    "grad_accum": "GRAD_ACCUM",
    "enc_len": "ENC_LEN",
    "epochs": "EPOCHS",
    "optim": "OPTIM",
    "eval_batch_size": "EVAL_BATCH_SIZE",
    "eval_iters": "EVAL_ITERS",
    "eval_multi_ckpt": "EVAL_MULTI_CKPT",
    "eval_intermediate_iters": "EVAL_INTERMEDIATE_ITERS",
    "eval_early_stop_small": "EVAL_EARLY_STOP_SMALL",
    "skip_data_gen": "SKIP_DATA_GEN",
    "skip_train": "SKIP_TRAIN",
    "skip_eval": "SKIP_EVAL",
    "save_only_at_end": "SAVE_ONLY_AT_END",
    "save_opt_state": "SAVE_OPT_STATE",
    "jax_compilation_cache_dir": "JAX_COMPILATION_CACHE_DIR",
    "shared_probe_dir": "SHARED_PROBE_DIR",
}


def env_value(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def merge_dicts(*dicts):
    result = {}
    for item in dicts:
        if item:
            result.update(item)
    return result


def slug_for_test_file(test_file):
    stem = Path(test_file).stem
    if stem.startswith("test_"):
        stem = stem[len("test_") :]
    return stem


def build_env(config, job, experiment, test_file):
    merged = merge_dicts(config.get("defaults"), job)
    env = os.environ.copy()

    for key, env_key in {**RESOURCE_KEYS, **EVAL_KEYS}.items():
        if key in merged and merged[key] is not None:
            env[env_key] = env_value(merged[key])

    if "spec_file" in job:
        env["SPEC_FILE"] = env_value(job["spec_file"])
        env.pop("EXPERIMENT_NAME", None)
    else:
        env["EXPERIMENT_NAME"] = experiment
        env.pop("SPEC_FILE", None)

    env["EVAL_TEST_FILES"] = test_file

    suffix_template = merged.get("job_name_suffix", "-eval-{test_slug}")
    env["JOB_NAME_SUFFIX"] = suffix_template.format(
        experiment=experiment,
        test_file=test_file,
        test_stem=Path(test_file).stem,
        test_slug=slug_for_test_file(test_file),
        job=job.get("name", "eval"),
    )

    return env


def iter_jobs(config):
    for job in config.get("jobs", []):
        experiments = job.get("experiments") or []
        if job.get("experiment"):
            experiments.append(job["experiment"])
        test_files = job.get("test_files") or []
        if job.get("test_file"):
            test_files.append(job["test_file"])

        if not experiments:
            raise ValueError("each job must define experiment(s)")
        if not test_files:
            raise ValueError("each job must define test_file(s)")

        for experiment in experiments:
            for test_file in test_files:
                if "," in test_file:
                    raise ValueError(
                        "test_files entries must be single files, not comma-separated values: %s"
                        % test_file
                    )
                yield job, experiment, test_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML eval submit config")
    parser.add_argument("--submit", action="store_true", help="actually call sbatch")
    parser.add_argument(
        "--submit-script",
        default="cluster/submit_attack_spec.sh",
        help="shell submit wrapper to invoke",
    )
    args = parser.parse_args()

    with open(args.config, "r") as file_in:
        config = yaml.safe_load(file_in) or {}

    submit_script = Path(args.submit_script)
    if not submit_script.is_file():
        raise SystemExit("submit script not found: %s" % submit_script)

    rows = []
    for job, experiment, test_file in iter_jobs(config):
        env = build_env(config, job, experiment, test_file)
        rows.append((job, experiment, test_file, env))

    if not rows:
        raise SystemExit("no jobs found in %s" % args.config)

    for _, experiment, test_file, env in rows:
        summary = "experiment=%s test_file=%s suffix=%s time=%s cpus=%s mem=%s gpu=%sx%s" % (
            experiment,
            test_file,
            env.get("JOB_NAME_SUFFIX", ""),
            env.get("TIME_LIMIT", ""),
            env.get("CPUS", ""),
            env.get("MEM", ""),
            env.get("GPU_TYPE", ""),
            env.get("GPU_COUNT", ""),
        )

        if not args.submit:
            print("DRY RUN", summary)
            continue

        proc = subprocess.run(
            ["bash", str(submit_script)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit(proc.returncode)
        print("%s job_id=%s" % (summary, proc.stdout.strip()))

    if not args.submit:
        print("\nDry run only. Re-run with --submit to submit %d jobs." % len(rows))


if __name__ == "__main__":
    main()
