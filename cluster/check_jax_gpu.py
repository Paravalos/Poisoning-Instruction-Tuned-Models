"""Print a small GPU/JAX diagnostic for Slurm sanity checks."""

from __future__ import annotations

import os
import platform


def main() -> None:
    print("python:", platform.python_version())
    print("hostname:", platform.node())
    print("cuda visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

    try:
        import jax
    except Exception as exc:
        print("jax import failed:", repr(exc))
        raise

    print("jax:", jax.__version__)
    print("jax backend:", jax.default_backend())
    print("jax process:", jax.process_index(), "/", jax.process_count())
    print("jax device count:", jax.device_count())
    for device in jax.devices():
        print("device:", device)


if __name__ == "__main__":
    main()
