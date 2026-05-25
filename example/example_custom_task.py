#!/usr/bin/env python3
"""
Example 2 - Custom task, built-in oracle policy.

env.reset() initialises the database and browser.  Afterwards, env.task is
replaced with BuyMostExpensiveInCategoryTask.  Because it subclasses
BuyCheapestInCategoryTask, scripted_oracle_actions() handles it automatically
via isinstance — no changes to the library needed.

Usage:
    python example/example_custom_task.py
    python example/example_custom_task.py --seed 99
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import db
from gym.env import EcommerceEnv
from gym.policies import scripted_oracle_actions
from gym.task import BuyCheapestInCategoryTask


class BuyMostExpensiveInCategoryTask(BuyCheapestInCategoryTask):
    """Custom task: buy the most expensive item in a randomly chosen category."""

    kind = "buy_most_expensive_in_category"

    def __init__(self, seed: int, db_path: Path, rng: random.Random) -> None:
        super().__init__(seed, db_path, rng)
        with db.connect(db_path) as conn:
            row = conn.execute(
                "SELECT sku FROM products WHERE category = ? "
                "ORDER BY price_cents DESC LIMIT 1",
                (self.category,),
            ).fetchone()
        self.expected_sku = row["sku"]

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.category.lower()}"

    @property
    def instruction(self) -> str:
        return (
            f"Buy the most expensive item in the '{self.category}' category "
            f"and ship it to {self.shipping_address}."
        )


def run_oracle_policy(env: EcommerceEnv, max_steps: int = 30) -> bool:
    """Drive an episode using the built-in scripted oracle."""
    success = False
    for step, action in enumerate(scripted_oracle_actions(env)[:max_steps], start=1):
        obs, reward, terminated, _, info = env.step(action)
        print(f"  step {step:2d}  url={obs['url']}  reward={reward:.1f}")
        success = reward > 0
        if terminated:
            break
    return success


def main(seed: int) -> None:
    print(f"=== Example: custom task, built-in oracle  seed={seed} ===")

    env = EcommerceEnv(seed=seed, headless=True)
    try:
        env.reset(seed=seed)

        # Inject custom task after reset.  Pass env.db_path so the task reads
        # the same seeded database the browser is connected to.
        # rng=random.Random(seed) is a fresh RNG, equivalent to what
        # AbstractEcommerceTask.sample() uses before advancing it for task
        # type selection — fine here since we construct the task directly.
        env.task = BuyMostExpensiveInCategoryTask(
            seed=seed, db_path=env.db_path, rng=random.Random(seed)
        )
        print(f"  task:        {env.task.kind}")
        print(f"  instruction: {env.task.instruction}")

        success = run_oracle_policy(env)
        print(f"  success: {success}")
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--seed", type=int, default=42, help="deterministic seed  [default: 42]")
    main(p.parse_args().seed)
