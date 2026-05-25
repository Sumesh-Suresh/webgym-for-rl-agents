from __future__ import annotations

import argparse
import random
from collections import defaultdict

from gym.debug_report import EpisodeDebugReport, render_episode_report, render_rollout_summary
from gym.env import EcommerceEnv
from gym.policies import random_action, scripted_oracle_actions


def run_episode(
    seed: int,
    policy: str,
    max_steps: int,
    debug: bool = False,
    continue_after_success: bool = False,
) -> tuple[str, str, bool, EpisodeDebugReport | None]:
    env = EcommerceEnv(seed=seed, headless=True, debug=False)
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
        else:
            rng = random.Random(seed)
            actions = [random_action(rng) for _ in range(max_steps)]

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run e-commerce gym episodes sequentially (one sampled task per episode)."
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--policy", choices=["oracle", "random"], default="oracle")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print a structured report for each episode.",
    )
    parser.add_argument(
        "--continue-after-success",
        action="store_true",
        help="Keep executing actions after the task verifier passes.",
    )
    args = parser.parse_args()

    aggregate: dict[str, list[bool]] = defaultdict(list)
    debug_reports: list[EpisodeDebugReport] = []
    for episode in range(args.episodes):
        seed = 10_000 + episode
        task_kind, task_name, success, report = run_episode(
            seed,
            args.policy,
            args.max_steps,
            args.debug,
            args.continue_after_success,
        )
        if report is not None:
            debug_reports.append(report)
        else:
            print(f"episode {episode} seed {seed}: {task_kind} ({task_name}) -> {success}")
        aggregate[task_kind].append(success)

    if debug_reports:
        print(f"\nrollout debug ({len(debug_reports)} episodes)\n")
        for report in debug_reports:
            print(render_episode_report(report))
            print()
        print(render_rollout_summary(debug_reports))

    for task_kind in sorted(aggregate):
        task_results = aggregate[task_kind]
        successes = sum(task_results)
        total = len(task_results)
        print(f"{task_kind}: {successes}/{total} success ({successes / total:.0%})")


if __name__ == "__main__":
    main()
