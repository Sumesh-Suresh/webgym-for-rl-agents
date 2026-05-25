from __future__ import annotations

from dataclasses import dataclass, field

from gym.actions import Action, Click, Navigate, Scroll, TypeText
from gym.ui_actions import (
    CancelOrder,
    FilterProducts,
    GoTo,
    OpenProduct,
    UiAction,
    UiClick,
    UiFill,
    ViewOrder,
)


def format_action(action: UiAction | Action) -> str:
    if isinstance(action, GoTo):
        return f"goto({action.page.value})"
    if isinstance(action, FilterProducts):
        parts = []
        if action.category.value:
            parts.append(f"category={action.category.value!r}")
        if action.sort.value != "alpha":
            parts.append(f"sort={action.sort.value}")
        args = ", ".join(parts) if parts else "defaults"
        return f"filter_products({args})"
    if isinstance(action, OpenProduct):
        return f"open_product({action.sku})"
    if isinstance(action, UiClick):
        return f"click({action.button.value})"
    if isinstance(action, UiFill):
        target = action.field.value
        if action.sku:
            target = f"{target}[{action.sku}]"
        return f"fill({target}, {action.value!r})"
    if isinstance(action, ViewOrder):
        return f"view_order({action.order_id})"
    if isinstance(action, CancelOrder):
        return f"cancel_order({action.order_id})"
    if isinstance(action, Click):
        return f"click({action.selector})"
    if isinstance(action, TypeText):
        return f"type({action.selector}, {action.text!r})"
    if isinstance(action, Scroll):
        return f"scroll({action.delta_y})"
    if isinstance(action, Navigate):
        return f"navigate({action.url})"
    return repr(action)


def _short_url(url: str, max_len: int = 48) -> str:
    if len(url) <= max_len:
        return url
    return url[: max_len - 1] + "…"


@dataclass
class StepRecord:
    step: int
    action: UiAction | Action
    reward: float
    terminated: bool
    url: str
    error: str = ""


@dataclass
class EpisodeDebugReport:
    seed: int
    task_kind: str
    task_name: str
    instruction: str
    policy: str
    planned_actions: list[UiAction | Action] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    success: bool = False

    def add_step(
        self,
        *,
        step: int,
        action: UiAction | Action,
        reward: float,
        terminated: bool,
        url: str,
        error: str = "",
    ) -> None:
        self.steps.append(
            StepRecord(
                step=step,
                action=action,
                reward=reward,
                terminated=terminated,
                url=url,
                error=error,
            )
        )
        if reward > 0:
            self.success = True


def render_episode_report(report: EpisodeDebugReport) -> str:
    width = 72
    title = f" seed {report.seed} · {report.task_kind} "
    bar = "─" * max(0, width - len(title))
    lines = [
        f"┌{title}{bar}",
        f"│ name: {report.task_name}",
        f"│ instruction: {report.instruction}",
    ]

    if report.planned_actions:
        lines.append("│")
        lines.append(f"│ planned ({report.policy}):")
        for index, action in enumerate(report.planned_actions, start=1):
            lines.append(f"│   {index:2}. {format_action(action)}")

    if report.steps:
        lines.append("│")
        lines.append("│ execution:")
        for record in report.steps:
            status = "done" if record.terminated else "    "
            reward = f"{record.reward:.1f}"
            action_text = format_action(record.action)
            url_text = _short_url(record.url)
            lines.append(
                f"│   {record.step:2}. {action_text:<28} r={reward:>3} {status}  {url_text}"
            )
            if record.error:
                lines.append(f"│       error: {record.error}")

    outcome = "PASS" if report.success else "FAIL"
    step_count = len(report.steps)
    lines.append("│")
    lines.append(f"│ result: {outcome} ({step_count} step{'s' if step_count != 1 else ''})")
    lines.append(f"└{'─' * (width - 1)}")
    return "\n".join(lines)


def render_rollout_summary(reports: list[EpisodeDebugReport]) -> str:
    if not reports:
        return ""

    lines = ["", "summary:"]
    for report in reports:
        outcome = "PASS" if report.success else "FAIL"
        lines.append(
            f"  {report.seed}  {outcome:<4}  {report.task_kind:<28}  {report.task_name}"
        )
    return "\n".join(lines)
