"""Deterministic seeding and RNG-state helpers.

CPU determinism is a project requirement: the same seed, config, data hashes and
code must produce the same step sequence. Accelerators are allowed to differ in
floating-point detail, which is recorded rather than hidden.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        # cuDNN autotuning would make repeated runs diverge; we prefer reproducible
        # kernels and accept the small throughput cost.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(_tuplify(state["python"]))
    if "numpy" in state:
        np_state = state["numpy"]
        np.random.set_state(
            (
                np_state[0],
                np.array(np_state[1], dtype=np.uint32),
                int(np_state[2]),
                int(np_state[3]),
                float(np_state[4]),
            )
        )
    if "torch" in state:
        tensor = state["torch"]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.uint8)
        torch.set_rng_state(tensor)
    if "cuda" in state and torch.cuda.is_available():
        states = [s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.uint8) for s in state["cuda"]]
        torch.cuda.set_rng_state_all(states)


def _tuplify(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuplify(v) for v in value)
    return value


def python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
