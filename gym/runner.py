from __future__ import annotations

import random

from gym.debug_report import EpisodeDebugReport
from gym.env import EcommerceEnv
from gym.policies import random_action, scripted_oracle_actions


def run_episode(
    seed: int,
    policy: str,
    max_steps: int,
    debug: bool = False,
    continue_after_success: bool = False,
) -> tuple[str, str, bool, EpisodeDebugReport | None]:
    """Run one episode and return (task_kind, task_name, success, optional_report).

    Args:
        seed: Deterministic seed for both database state and task selection.
        policy: "oracle" for the scripted oracle, "random" for random navigation.
        max_steps: Maximum number of steps before the episode is cut short.
        debug: When True, build and return a full EpisodeDebugReport.
        continue_after_success: When True, keep stepping even after the verifier passes.
    """
    env = EcommerceEnv(seed=seed, headless=True)
    report: EpisodeDebugReport | None = None
    try:
        observation, reset_info = env.reset(seed=seed)
        task_kind = reset_info["task_kind"]
        task_name = reset_info["task_name"]

        if debug:
            report = EpisodeDebugReport(
                seed=seed,
                task_kind=task_kind,
                task_name=task_name,
                instruction=reset_info["instruction"],
                policy=policy,
            )

        if policy == "oracle":
            actions = scripted_oracle_actions(env)
            if report is not None:
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
            if report is not None:
                report.add_step(
                    step=step,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    url=observation["url"],
                    error=str(info.get("last_action_error") or ""),
                )
            if terminated and not continue_after_success:
                break

        if report is not None and not report.success:
            report.success = success
        return task_kind, task_name, success, report
    finally:
        env.close()
