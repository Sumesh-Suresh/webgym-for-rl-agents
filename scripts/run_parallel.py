"""Parallel-rollout demo for the e-commerce gym.

Launches `--workers` processes, each spinning up its own EcomEnv (own port, own
SQLite file, own browser context), and runs a fixed number of episodes per task
under both the scripted-oracle policy and the random-clicks baseline policy.

Prints per-(task × policy) success rates and total wall time. The point is to
demonstrate two things:

  1) Concurrent envs don't share mutable state. If they did, you'd see
     cross-contamination — e.g., the cancel-recent verifier passing because
     a *different* worker cancelled its own order against the same DB. The
     fact that random-policy success rates stay near zero across runs is a
     positive signal here.

  2) Oracles pass at high rates, which makes the verifiers' "what does
     success look like" intent legible without running an actual agent.

Usage:
    python -m scripts.run_parallel --workers 4 --episodes 2
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from gym.env import EcomEnv
from gym.tasks import TASKS
from scripts.policies import RandomPolicy


@dataclass
class EpisodeResult:
    task_id: str
    policy: str
    worker_id: int
    episode: int
    success: bool
    steps: int
    duration_s: float
    reason: str = ""


def run_oracle_episode(task_id: str, worker_id: int, episode: int) -> EpisodeResult:
    """One episode driven by the task's scripted oracle. The oracle drives the
    Playwright page directly; we then call env.step({'type': 'noop'}) once so
    the verifier runs and we get a standard (obs, reward, terminated, ...) tuple."""
    t0 = time.monotonic()
    env = EcomEnv(task_id=task_id, max_steps=5, headless=True, screenshot=False)
    try:
        env.reset(seed=worker_id)
        TASKS[task_id].oracle(env.page)
        _, reward, terminated, _truncated, info = env.step({"type": "noop"})
        return EpisodeResult(
            task_id=task_id,
            policy="oracle",
            worker_id=worker_id,
            episode=episode,
            success=bool(terminated),
            steps=1,
            duration_s=time.monotonic() - t0,
            reason="" if reward > 0 else info.get("verify", {}).get("reason", ""),
        )
    finally:
        env.close()


def run_random_episode(task_id: str, worker_id: int, episode: int, max_steps: int = 25) -> EpisodeResult:
    t0 = time.monotonic()
    env = EcomEnv(task_id=task_id, max_steps=max_steps, headless=True, screenshot=False)
    policy = RandomPolicy(seed=(worker_id * 1000 + episode))
    try:
        obs, _info = env.reset(seed=worker_id)
        steps = 0
        terminated = truncated = False
        reason = ""
        while not (terminated or truncated):
            action = policy.act(obs)
            obs, _reward, terminated, truncated, info = env.step(action)
            steps += 1
            reason = info.get("verify", {}).get("reason", reason)
        return EpisodeResult(
            task_id=task_id,
            policy="random",
            worker_id=worker_id,
            episode=episode,
            success=terminated,
            steps=steps,
            duration_s=time.monotonic() - t0,
            reason="" if terminated else reason,
        )
    finally:
        env.close()


def _dispatch(spec):
    task_id, policy, worker_id, episode = spec
    if policy == "oracle":
        return run_oracle_episode(task_id, worker_id, episode)
    return run_random_episode(task_id, worker_id, episode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=2,
                        help="Episodes per (task, policy) per worker")
    parser.add_argument("--policies", default="oracle,random",
                        help="Comma-separated subset of {oracle, random}")
    parser.add_argument("--tasks", default=",".join(TASKS),
                        help="Comma-separated task subset")
    args = parser.parse_args()

    policies = [p for p in args.policies.split(",") if p]
    task_ids = [t for t in args.tasks.split(",") if t]
    for t in task_ids:
        if t not in TASKS:
            raise SystemExit(f"unknown task: {t}")
    for p in policies:
        if p not in {"oracle", "random"}:
            raise SystemExit(f"unknown policy: {p}")

    work = []
    for worker_id in range(args.workers):
        for task_id in task_ids:
            for policy in policies:
                for ep in range(args.episodes):
                    work.append((task_id, policy, worker_id, ep))

    print(f"running {len(work)} episodes across {args.workers} workers...")
    t0 = time.monotonic()
    results: list[EpisodeResult] = []
    # max_workers caps concurrency; each EpisodeResult is independent.
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_dispatch, spec) for spec in work]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            flag = "✓" if r.success else "✗"
            print(
                f"  {flag} task={r.task_id:32s} policy={r.policy:6s} "
                f"worker={r.worker_id} ep={r.episode} steps={r.steps:3d} "
                f"t={r.duration_s:5.2f}s "
                + (f"reason={r.reason}" if r.reason and not r.success else "")
            )

    wall = time.monotonic() - t0
    print(f"\nwall time: {wall:.1f}s")
    print("\nsuccess rates:")
    print(f"  {'task':32s} {'policy':8s} {'success':>10s} {'avg_steps':>10s}")
    for task_id in task_ids:
        for policy in policies:
            subset = [r for r in results if r.task_id == task_id and r.policy == policy]
            if not subset:
                continue
            n_ok = sum(1 for r in subset if r.success)
            avg_steps = sum(r.steps for r in subset) / len(subset)
            print(
                f"  {task_id:32s} {policy:8s} {n_ok}/{len(subset):<8} {avg_steps:>10.1f}"
            )


if __name__ == "__main__":
    main()
