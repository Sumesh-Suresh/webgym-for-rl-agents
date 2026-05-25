"""Rollout tests: oracle policy should pass validation; random policy should not."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import pytest

from gym.debug_report import EpisodeDebugReport, render_episode_report, render_rollout_summary
from gym.env import EcommerceEnv
from gym.policies import random_action, scripted_oracle_actions

_REPO_ROOT = Path(__file__).resolve().parent
_LOCAL_PLAYWRIGHT = _REPO_ROOT / ".playwright"
if _LOCAL_PLAYWRIGHT.is_dir() and "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(_LOCAL_PLAYWRIGHT)

MAX_STEPS = 12

# Four deterministic seeds per task kind (task sampling is seed-driven).
ORACLE_CASES: list[tuple[str, int]] = [
    ("buy_cheapest_in_category", 1),
    ("buy_cheapest_in_category", 2),
    ("buy_cheapest_in_category", 3),
    ("buy_cheapest_in_category", 4),
    ("apply_coupon_with_quantity", 7),
    ("apply_coupon_with_quantity", 9),
    ("apply_coupon_with_quantity", 11),
    ("apply_coupon_with_quantity", 12),
    ("cancel_recent_order", 5),
    ("cancel_recent_order", 6),
    ("cancel_recent_order", 10),
    ("cancel_recent_order", 17),
]

# Seeds chosen so random navigation does not accidentally satisfy verifiers.
RANDOM_CASES: list[tuple[str, int]] = [
    ("buy_cheapest_in_category", 100),
    ("cancel_recent_order", 101),
    ("apply_coupon_with_quantity", 109),
    ("apply_coupon_with_quantity", 110),
]


@dataclass(frozen=True)
class RolloutTestResult:
    report: EpisodeDebugReport
    validation_pass: bool

    @property
    def policy_pass(self) -> bool:
        return self.report.success

    @property
    def policy_outcome(self) -> str:
        return "PASS" if self.policy_pass else "FAIL"

    @property
    def validation_outcome(self) -> str:
        return "PASS" if self.validation_pass else "FAIL"


def run_episode(
    seed: int,
    policy: str,
    *,
    max_steps: int = MAX_STEPS,
) -> RolloutTestResult:
    env = EcommerceEnv(seed=seed, headless=True, debug=False)
    validation_pass = False
    try:
        observation, reset_info = env.reset(seed=seed)
        report = EpisodeDebugReport(
            seed=seed,
            task_kind=reset_info["task_kind"],
            task_name=reset_info["task_name"],
            instruction=reset_info["instruction"],
            policy=policy,
        )

        if policy == "oracle":
            actions = scripted_oracle_actions(env)
            report.planned_actions = list(actions)
        elif policy == "random":
            rng = random.Random(seed)
            actions = [random_action(rng) for _ in range(max_steps)]
        else:
            raise ValueError(f"unknown policy: {policy!r}")

        success = False
        for step, action in enumerate(actions[:max_steps], start=1):
            observation, reward, terminated, _, info = env.step(action)
            success = reward > 0
            step_validation_pass = bool(info.get("task_info", {}).get("success", False))
            validation_pass = validation_pass or step_validation_pass
            report.add_step(
                step=step,
                action=action,
                reward=reward,
                terminated=terminated,
                url=observation["url"],
                error=str(info.get("last_action_error") or ""),
            )
            if terminated:
                break

        if not report.success:
            report.success = success

        return RolloutTestResult(report=report, validation_pass=validation_pass)
    finally:
        env.close()


def _assert_outcomes(
    result: RolloutTestResult,
    *,
    expected_task_kind: str,
    expect_pass: bool,
) -> None:
    report = result.report
    assert report.task_kind == expected_task_kind, (
        f"seed {report.seed}: expected task {expected_task_kind!r}, "
        f"got {report.task_kind!r}"
    )

    policy_ok = result.policy_pass == expect_pass
    validation_ok = result.validation_pass == expect_pass
    assert policy_ok, (
        f"seed {report.seed} ({report.task_kind}, {report.policy}): "
        f"policy {result.policy_outcome}, expected "
        f"{'PASS' if expect_pass else 'FAIL'}"
    )
    assert validation_ok, (
        f"seed {report.seed} ({report.task_kind}, {report.policy}): "
        f"validation {result.validation_outcome}, expected "
        f"{'PASS' if expect_pass else 'FAIL'}"
    )
    assert result.policy_pass == result.validation_pass, (
        f"seed {report.seed}: policy {result.policy_outcome} != "
        f"validation {result.validation_outcome}"
    )


@pytest.mark.parametrize("task_kind,seed", ORACLE_CASES)
def test_oracle_policy_passes_for_all_tasks(task_kind: str, seed: int) -> None:
    result = run_episode(seed, "oracle")
    _assert_outcomes(result, expected_task_kind=task_kind, expect_pass=True)


@pytest.mark.parametrize("task_kind,seed", RANDOM_CASES)
def test_random_policy_fails_for_all_tasks(task_kind: str, seed: int) -> None:
    result = run_episode(seed, "random")
    _assert_outcomes(result, expected_task_kind=task_kind, expect_pass=False)


def test_oracle_and_random_rollout_matrix() -> None:
    """Run all 16 cases and print debug reports plus a combined summary."""
    results: list[RolloutTestResult] = []

    for task_kind, seed in ORACLE_CASES:
        result = run_episode(seed, "oracle")
        _assert_outcomes(result, expected_task_kind=task_kind, expect_pass=True)
        results.append(result)

    for task_kind, seed in RANDOM_CASES:
        result = run_episode(seed, "random")
        _assert_outcomes(result, expected_task_kind=task_kind, expect_pass=False)
        results.append(result)

    print("\nrollout_test debug (16 episodes)\n")
    for result in results:
        report = result.report
        print(render_episode_report(report))
        print(
            f"│ validation: {result.validation_outcome} "
            f"(policy: {result.policy_outcome})"
        )
        print()

    print(render_rollout_summary([result.report for result in results]))
    print()
    for result in results:
        report = result.report
        print(
            f"  {report.seed:>5}  policy={result.policy_outcome:<4}  "
            f"validation={result.validation_outcome:<4}  "
            f"{report.policy:<6}  {report.task_kind}"
        )


if __name__ == "__main__":
    test_oracle_and_random_rollout_matrix()
    print("\nall rollout tests passed")
