#!/usr/bin/env python3
"""
Example 1 — No customization.

run_episode() handles environment setup, oracle policy execution, and teardown.
Use this when you only need the aggregate result, not per-step control.

Usage:
    python example/example_oracle.py
    python example/example_oracle.py --seed 99
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gym.runner import run_episode


def main(seed: int) -> None:
    print(f"=== Example: oracle (no customization)  seed={seed} ===")

    task_kind, task_name, success, _ = run_episode(
        seed=seed,
        policy="oracle",
        max_steps=20,
    )
    print(f"  task={task_kind}  name={task_name}  success={success}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42, help="deterministic seed  [default: 42]")
    main(p.parse_args().seed)
