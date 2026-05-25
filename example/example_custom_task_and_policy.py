#!/usr/bin/env python3
"""
Example 3 - Custom task + custom policy.

This script replaces env.task with BuyMostExpensiveInCategoryTask after reset,
then drives that task with a custom policy that returns a concrete list of
actions.

Usage:
    python example/example_custom_task_and_policy.py
    python example/example_custom_task_and_policy.py --seed 7
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import db
from gym.env import EcommerceEnv
from gym.task import BuyCheapestInCategoryTask
from gym.ui_actions import (
    FormField,
    UiAction,
    UiButton,
    click_ui,
    fill_field,
    open_product,
)


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


def custom_policy(task: BuyMostExpensiveInCategoryTask) -> list[UiAction]:
    """Return the exact action list needed to complete the custom task."""
    return [
        open_product(task.expected_sku),
        click_ui(UiButton.ADD_TO_CART),
        click_ui(UiButton.CHECKOUT_LINK),
        fill_field(FormField.SHIPPING_ADDRESS, task.shipping_address),
        click_ui(UiButton.PLACE_ORDER),
    ]


def run_action_list(env: EcommerceEnv, actions: list[UiAction]) -> bool:
    """Drive an episode using a precomputed action list."""
    success = False
    for step, action in enumerate(actions, start=1):
        obs, reward, terminated, _, info = env.step(action)
        print(f"  step {step:2d}  url={obs['url']}  reward={reward:.1f}")
        success = reward > 0
        if terminated:
            break
    return success


def main(seed: int) -> None:
    print(f"=== Example: custom task + custom policy  seed={seed} ===")

    env = EcommerceEnv(seed=seed, headless=True)
    try:
        env.reset(seed=seed)

        env.task = BuyMostExpensiveInCategoryTask(
            seed=seed, db_path=env.db_path, rng=random.Random(seed)
        )
        print(f"  task:        {env.task.kind}")
        print(f"  instruction: {env.task.instruction}")

        success = run_action_list(env, custom_policy(env.task))
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
