"""
End-to-end environment tests covering oracle, random, random-then-oracle,
and adversarial scenarios designed to expose detection failures.

Detection logic
---------------
  Oracle rewarded                 → CORRECT  (task completed as expected)
  Oracle not rewarded             → WRONG    (oracle failed — env or policy bug)
  Random rewarded                 → WRONG    (random accidentally passed a verifier)
  Random not rewarded             → CORRECT  (env correctly withheld reward)
  Random-then-oracle rewarded     → CORRECT  (oracle recovered after random warmup)
  Random-then-oracle not rewarded → WRONG    (oracle failed after random prefix)
  Adversarial rewarded            → WRONG    (verifier too loose — false positive)
  Adversarial not rewarded        → CORRECT  (verifier correctly rejected near-miss)

Adversarial cases probe every axis the verifiers check:
  • cancel: wrong target order, newly-placed-then-cancelled order
  • buy:    wrong SKU, wrong shipping address, extra item in cart (2-item order)
  • coupon: wrong quantity, missing coupon code
  • env:    step budget too small, cart state leak across resets

Run with:
    .venv/bin/pytest tests/rollout_test_e2e.py -v -s
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from app import db
from gym.actions import Action, Click, Navigate, TypeText
from gym.debug_report import EpisodeDebugReport, render_episode_report
from gym.env import EcommerceEnv
from gym.policies import random_action, scripted_oracle_actions
from gym.runner import run_episode
from gym.ui_actions import UiAction

# ── Browser / playwright setup ────────────────────────────────────────────────

_LOCAL_PLAYWRIGHT = _REPO_ROOT / ".playwright"
if _LOCAL_PLAYWRIGHT.is_dir() and "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(_LOCAL_PLAYWRIGHT)

MAX_STEPS = 15

# ── Seeds (verified empirically after DB-seeding changes) ─────────────────────
#
#   CANCEL_SEED  → cancel_recent_order        (2 placed orders: target=2, other=1)
#   BUY_SEED     → buy_cheapest_in_category   (Electronics, cheapest=SKU-E1001)
#   COUPON_SEED  → apply_coupon_with_quantity  (SKU-H2002, qty=1, coupon=SAVE10)

CANCEL_SEED  = 5
BUY_SEED     = 1
COUPON_SEED  = 0

ORACLE_SEEDS  = [0, 7, 1, 2, 5, 6]   # two seeds per task kind
RANDOM_SEEDS  = [200, 201, 202, 203]  # far from training range
PARTIAL_SEEDS = [7, 2, 6]            # one per task kind for random_then_oracle

PolicyKind = Literal["oracle", "random", "random_then_oracle",
                     "adv_cancel_wrong_order", "adv_cancel_new_order",
                     "adv_buy_wrong_sku", "adv_buy_wrong_address",
                     "adv_buy_extra_cart_item",
                     "adv_coupon_wrong_quantity", "adv_coupon_no_coupon",
                     "adv_budget_exhausted"]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class E2EResult:
    seed: int
    task_kind: str
    policy: PolicyKind
    success: bool
    expected_success: bool
    steps: int
    report: EpisodeDebugReport | None = None

    @property
    def detection(self) -> str:
        return "CORRECT" if self.success == self.expected_success else "WRONG"

    @property
    def outcome(self) -> str:
        return "PASS" if self.success else "FAIL"


# ── Standard policy runners ───────────────────────────────────────────────────

def run_oracle(seed: int) -> E2EResult:
    task_kind, _, success, report = run_episode(seed, "oracle", MAX_STEPS, debug=True)
    return E2EResult(seed=seed, task_kind=task_kind, policy="oracle",
                     success=success, expected_success=True,
                     steps=len(report.steps) if report else 0, report=report)


def run_random(seed: int) -> E2EResult:
    task_kind, _, success, report = run_episode(seed, "random", MAX_STEPS, debug=True)
    return E2EResult(seed=seed, task_kind=task_kind, policy="random",
                     success=success, expected_success=False,
                     steps=len(report.steps) if report else 0, report=report)


def run_random_then_oracle(seed: int, random_prefix: int = 4) -> E2EResult:
    """random_prefix random steps, then oracle sequence for the remaining budget."""
    env = EcommerceEnv(seed=seed, headless=True)
    rng = random.Random(seed ^ 0xBEEF)
    report = EpisodeDebugReport(seed=seed, task_kind="", task_name="",
                                instruction="", policy="random_then_oracle")
    try:
        obs, info = env.reset(seed=seed)
        report.task_kind = info["task_kind"]
        report.task_name = info["task_name"]
        report.instruction = info["instruction"]
        step, success = 0, False

        for _ in range(random_prefix):
            if step >= MAX_STEPS:
                break
            act = random_action(rng)
            obs, reward, terminated, _, info = env.step(act)
            step += 1
            report.add_step(step=step, action=act, reward=reward, terminated=terminated,
                            url=obs["url"], error=str(info.get("last_action_error") or ""))
            if terminated:
                success = reward > 0
                break

        if not success and step < MAX_STEPS:
            oracle_actions = scripted_oracle_actions(env)
            report.planned_actions = list(oracle_actions)
            for act in oracle_actions:
                if step >= MAX_STEPS:
                    break
                obs, reward, terminated, _, info = env.step(act)
                step += 1
                report.add_step(step=step, action=act, reward=reward, terminated=terminated,
                                url=obs["url"], error=str(info.get("last_action_error") or ""))
                if terminated:
                    success = reward > 0
                    break

        report.success = success
        return E2EResult(seed=seed, task_kind=report.task_kind, policy="random_then_oracle",
                         success=success, expected_success=True, steps=step, report=report)
    finally:
        env.close()


# ── Adversarial runners ───────────────────────────────────────────────────────
# Each scenario targets a specific verifier axis.  All are expected to yield
# reward = 0 (expected_success=False).  A WRONG detection means the verifier
# accepted something it should have rejected — a false positive.

def _last_reward(env: EcommerceEnv, actions: list[Action]) -> tuple[float, int]:
    """Execute a sequence of actions; return (final_reward, step_count)."""
    reward, step = 0.0, 0
    for act in actions:
        _, reward, terminated, _, _ = env.step(act)
        step += 1
        if terminated:
            break
    return reward, step


def run_adv_cancel_wrong_order() -> E2EResult:
    """Cancel a PLACED order that is NOT the anchored target.

    Verifier uses target_order_id; cancelling a different order must not satisfy it.
    Failure mode: verifier checks 'any cancelled order' instead of the specific target.
    """
    env = EcommerceEnv(seed=CANCEL_SEED, headless=True)
    try:
        _, info = env.reset(seed=CANCEL_SEED)
        task = env.task
        other_placed = [o for o in db.list_orders(env.db_path)
                        if o["status"] == "placed" and int(o["id"]) != task.target_order_id]
        assert other_placed, f"CANCEL_SEED={CANCEL_SEED} has no other placed order — update seed"
        other_id = int(other_placed[0]["id"])
        reward, steps = _last_reward(env, [
            Navigate("/orders"),
            Click(f"[data-testid='cancel-order-{other_id}']"),
        ])
        return E2EResult(seed=CANCEL_SEED, task_kind=info["task_kind"],
                         policy="adv_cancel_wrong_order",
                         success=reward > 0, expected_success=False, steps=steps)
    finally:
        env.close()


def run_adv_cancel_new_order() -> E2EResult:
    """Place a brand-new order mid-episode and cancel it; leave the seeded target untouched.

    Verifier anchors to target_order_id at reset; new orders must not shift the target.
    Failure mode: verifier uses 'most recent cancelled order' logic instead of a fixed ID.
    """
    env = EcommerceEnv(seed=CANCEL_SEED, headless=True)
    try:
        _, info = env.reset(seed=CANCEL_SEED)
        task = env.task
        some_sku = db.list_products(env.db_path)[0]["sku"]

        # Place a new order through the browser
        env.step(Navigate(f"/product/{some_sku}"))
        env.step(Click("[data-testid='add-to-cart']"))
        env.step(Navigate("/checkout"))
        env.step(TypeText("[data-testid='shipping-address']", "1 Test Ave, Chicago, IL 60601"))
        env.step(Click("[data-testid='place-order']"))

        # The new order is the most recent — cancel it
        new_order_id = int(db.list_orders(env.db_path)[0]["id"])
        assert new_order_id != task.target_order_id
        env.step(Navigate("/orders"))
        reward, steps = _last_reward(env, [
            Click(f"[data-testid='cancel-order-{new_order_id}']"),
        ])
        return E2EResult(seed=CANCEL_SEED, task_kind=info["task_kind"],
                         policy="adv_cancel_new_order",
                         success=reward > 0, expected_success=False, steps=steps + 6)
    finally:
        env.close()


def run_adv_buy_wrong_sku() -> E2EResult:
    """Checkout the second-cheapest item in the category instead of the cheapest.

    Verifier checks order["sku"] == expected_sku; a more expensive item must be rejected.
    Failure mode: verifier only checks category or price range, not the specific SKU.
    """
    env = EcommerceEnv(seed=BUY_SEED, headless=True)
    try:
        _, info = env.reset(seed=BUY_SEED)
        task = env.task
        # Find any product in the same category that is NOT the cheapest
        prods = db.list_products(env.db_path, category=task.category, sort_by="price_asc")
        wrong_sku = next(p["sku"] for p in prods if p["sku"] != task.expected_sku)
        reward, steps = _last_reward(env, [
            Navigate(f"/product/{wrong_sku}"),
            Click("[data-testid='add-to-cart']"),
            Navigate("/checkout"),
            TypeText("[data-testid='shipping-address']", task.shipping_address),
            Click("[data-testid='place-order']"),
        ])
        return E2EResult(seed=BUY_SEED, task_kind=info["task_kind"],
                         policy="adv_buy_wrong_sku",
                         success=reward > 0, expected_success=False, steps=steps)
    finally:
        env.close()


def run_adv_buy_wrong_address() -> E2EResult:
    """Checkout the correct cheapest SKU but ship it to a wrong address.

    Verifier checks order["shipping_address"] == task.shipping_address.
    Failure mode: verifier checks SKU but not the shipping destination.
    """
    env = EcommerceEnv(seed=BUY_SEED, headless=True)
    try:
        _, info = env.reset(seed=BUY_SEED)
        task = env.task
        wrong_addr = next(a for a in db.ADDRESS_LIST if a != task.shipping_address)
        reward, steps = _last_reward(env, [
            Navigate(f"/product/{task.expected_sku}"),
            Click("[data-testid='add-to-cart']"),
            Navigate("/checkout"),
            TypeText("[data-testid='shipping-address']", wrong_addr),
            Click("[data-testid='place-order']"),
        ])
        return E2EResult(seed=BUY_SEED, task_kind=info["task_kind"],
                         policy="adv_buy_wrong_address",
                         success=reward > 0, expected_success=False, steps=steps)
    finally:
        env.close()


def run_adv_buy_extra_cart_item() -> E2EResult:
    """Add a wrong item to the cart first, then let oracle add the correct item and check out.

    The resulting order has 2 line items.  Verifier requires len(items) == 1.
    Failure mode: verifier finds the correct SKU in items[] but ignores extra items.
    """
    env = EcommerceEnv(seed=BUY_SEED, headless=True)
    try:
        _, info = env.reset(seed=BUY_SEED)
        task = env.task
        wrong_sku = next(p["sku"] for p in db.list_products(env.db_path)
                         if p["sku"] != task.expected_sku)
        # Contaminate cart with wrong item
        env.step(Navigate(f"/product/{wrong_sku}"))
        env.step(Click("[data-testid='add-to-cart']"))

        # Oracle adds the correct item and checks out — cart has 2 items at checkout
        step = 2
        reward = 0.0
        for act in scripted_oracle_actions(env):
            _, reward, terminated, _, _ = env.step(act)
            step += 1
            if terminated:
                break
        return E2EResult(seed=BUY_SEED, task_kind=info["task_kind"],
                         policy="adv_buy_extra_cart_item",
                         success=reward > 0, expected_success=False, steps=step)
    finally:
        env.close()


def run_adv_coupon_wrong_quantity() -> E2EResult:
    """Add task.quantity + 1 units (instead of the required quantity) with the correct coupon.

    Verifier checks items[0]["quantity"] == task.quantity exactly.
    Failure mode: verifier checks SKU and coupon but uses >= instead of == for quantity.
    """
    env = EcommerceEnv(seed=COUPON_SEED, headless=True)
    try:
        _, info = env.reset(seed=COUPON_SEED)
        task = env.task
        wrong_qty = task.quantity + 1
        reward, steps = _last_reward(env, [
            Navigate(f"/product/{task.sku}"),
            TypeText("[data-testid='quantity']", str(wrong_qty)),
            Click("[data-testid='add-to-cart']"),
            TypeText("[data-testid='coupon-code']", task.coupon_code),
            Click("[data-testid='apply-coupon']"),
            Click("[data-testid='checkout-link']"),
            TypeText("[data-testid='shipping-address']", task.shipping_address),
            Click("[data-testid='place-order']"),
        ])
        return E2EResult(seed=COUPON_SEED, task_kind=info["task_kind"],
                         policy="adv_coupon_wrong_quantity",
                         success=reward > 0, expected_success=False, steps=steps)
    finally:
        env.close()


def run_adv_coupon_no_coupon() -> E2EResult:
    """Checkout the correct SKU and quantity but skip applying the coupon entirely.

    Verifier checks order["coupon_code"] == task.coupon_code.
    Failure mode: verifier checks SKU + quantity but not whether a coupon was applied.
    """
    env = EcommerceEnv(seed=COUPON_SEED, headless=True)
    try:
        _, info = env.reset(seed=COUPON_SEED)
        task = env.task
        reward, steps = _last_reward(env, [
            Navigate(f"/product/{task.sku}"),
            TypeText("[data-testid='quantity']", str(task.quantity)),
            Click("[data-testid='add-to-cart']"),
            # No coupon applied
            Click("[data-testid='checkout-link']"),
            TypeText("[data-testid='shipping-address']", task.shipping_address),
            Click("[data-testid='place-order']"),
        ])
        return E2EResult(seed=COUPON_SEED, task_kind=info["task_kind"],
                         policy="adv_coupon_no_coupon",
                         success=reward > 0, expected_success=False, steps=steps)
    finally:
        env.close()


def run_adv_budget_exhausted() -> E2EResult:
    """Run the oracle for the buy task with only 2 steps — not enough to complete it.

    buy_cheapest needs 6 oracle steps; stopping after 2 leaves no order in the DB.
    Failure mode: env grants partial credit or verifier fires on incomplete state.
    """
    task_kind, _, success, report = run_episode(BUY_SEED, "oracle", max_steps=2, debug=True)
    return E2EResult(seed=BUY_SEED, task_kind=task_kind, policy="adv_budget_exhausted",
                     success=success, expected_success=False,
                     steps=len(report.steps) if report else 0, report=report)


# ── Individual parametrized assertions ────────────────────────────────────────

@pytest.mark.parametrize("seed", ORACLE_SEEDS)
def test_oracle_is_rewarded(seed: int) -> None:
    result = run_oracle(seed)
    assert result.success, (
        f"seed={seed} task={result.task_kind}: oracle NOT rewarded — detection WRONG\n"
        + (render_episode_report(result.report) if result.report else "")
    )


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_random_is_not_rewarded(seed: int) -> None:
    result = run_random(seed)
    assert not result.success, (
        f"seed={seed} task={result.task_kind}: random accidentally REWARDED — detection WRONG\n"
        + (render_episode_report(result.report) if result.report else "")
    )


@pytest.mark.parametrize("seed", PARTIAL_SEEDS)
def test_random_then_oracle_is_rewarded(seed: int) -> None:
    result = run_random_then_oracle(seed)
    assert result.success, (
        f"seed={seed} task={result.task_kind}: random_then_oracle NOT rewarded — detection WRONG\n"
        + (render_episode_report(result.report) if result.report else "")
    )


def test_cancel_wrong_order_not_rewarded() -> None:
    result = run_adv_cancel_wrong_order()
    assert not result.success, "detection WRONG: cancelling non-target order gave reward"


def test_cancel_new_order_not_rewarded() -> None:
    result = run_adv_cancel_new_order()
    assert not result.success, "detection WRONG: cancelling a newly-placed order gave reward"


def test_buy_wrong_sku_not_rewarded() -> None:
    result = run_adv_buy_wrong_sku()
    assert not result.success, "detection WRONG: wrong SKU checkout gave reward"


def test_buy_wrong_address_not_rewarded() -> None:
    result = run_adv_buy_wrong_address()
    assert not result.success, "detection WRONG: wrong shipping address gave reward"


def test_buy_extra_item_in_cart_not_rewarded() -> None:
    result = run_adv_buy_extra_cart_item()
    assert not result.success, "detection WRONG: 2-item checkout (cart contamination) gave reward"


def test_apply_coupon_wrong_quantity_not_rewarded() -> None:
    result = run_adv_coupon_wrong_quantity()
    assert not result.success, "detection WRONG: wrong quantity with coupon gave reward"


def test_apply_coupon_no_coupon_not_rewarded() -> None:
    result = run_adv_coupon_no_coupon()
    assert not result.success, "detection WRONG: missing coupon gave reward"


def test_oracle_budget_exhausted_not_rewarded() -> None:
    result = run_adv_budget_exhausted()
    assert not result.success, "detection WRONG: incomplete oracle (2 steps) gave reward"


def test_reset_clears_cart_between_episodes() -> None:
    """Cart state (cookie-based) must not leak from one episode to the next.

    After episode 1 adds items to the cart, env.reset() must produce a fresh browser
    context with an empty cart.  A leaked cart would corrupt subsequent task verifiers.
    """
    env = EcommerceEnv(seed=BUY_SEED, headless=True)
    try:
        # Episode 1: add an item to the cart
        env.reset(seed=BUY_SEED)
        task = env.task
        env.step(Navigate(f"/product/{task.expected_sku}"))
        env.step(Click("[data-testid='add-to-cart']"))
        obs, _, _, _, _ = env.step(Navigate("/cart"))
        assert "Your cart is empty" not in obs["dom"], "cart should have item in episode 1"

        # Episode 2: reset to a different seed
        env.reset(seed=COUPON_SEED)
        obs, _, _, _, _ = env.step(Navigate("/cart"))
        assert "Your cart is empty" in obs["dom"], (
            "cart state leaked into episode 2 — reset did not clear browser context"
        )
    finally:
        env.close()


# ── Full matrix with printed report ──────────────────────────────────────────

def test_e2e_full_report() -> None:
    """Run all policy variants (including adversarial) and print a detection report."""
    results: list[E2EResult] = []

    for seed in ORACLE_SEEDS:
        results.append(run_oracle(seed))
    for seed in RANDOM_SEEDS:
        results.append(run_random(seed))
    for seed in PARTIAL_SEEDS:
        results.append(run_random_then_oracle(seed))

    results += [
        run_adv_cancel_wrong_order(),
        run_adv_cancel_new_order(),
        run_adv_buy_wrong_sku(),
        run_adv_buy_wrong_address(),
        run_adv_buy_extra_cart_item(),
        run_adv_coupon_wrong_quantity(),
        run_adv_coupon_no_coupon(),
        run_adv_budget_exhausted(),
    ]

    _print_report(results)

    wrong = [r for r in results if r.detection == "WRONG"]
    assert not wrong, (
        f"{len(wrong)} wrong detection(s):\n"
        + "\n".join(f"  seed={r.seed} policy={r.policy} outcome={r.outcome}" for r in wrong)
    )


# ── Report renderer ───────────────────────────────────────────────────────────

_POLICY_GROUPS = [
    ("oracle",               "Oracle (expect: rewarded)"),
    ("random",               "Random (expect: not rewarded)"),
    ("random_then_oracle",   "Random → Oracle (expect: rewarded)"),
    ("adv_cancel_wrong_order",   "Adversarial — cancel task"),
    ("adv_cancel_new_order",     "Adversarial — cancel task"),
    ("adv_buy_wrong_sku",        "Adversarial — buy task"),
    ("adv_buy_wrong_address",    "Adversarial — buy task"),
    ("adv_buy_extra_cart_item",  "Adversarial — buy task"),
    ("adv_coupon_wrong_quantity","Adversarial — coupon task"),
    ("adv_coupon_no_coupon",     "Adversarial — coupon task"),
    ("adv_budget_exhausted",     "Adversarial — env"),
]


def _print_report(results: list[E2EResult]) -> None:
    col_w = [6, 30, 26, 6, 9, 9]
    header = ["SEED", "TASK KIND", "POLICY", "STEPS", "OUTCOME", "DETECTION"]
    sep = "  ".join("─" * w for w in col_w)

    by_policy = {r.policy: r for r in results}   # one result per policy key (adversarial)
    by_policy_list: dict[str, list[E2EResult]] = {}
    for r in results:
        by_policy_list.setdefault(r.policy, []).append(r)

    lines: list[str] = ["", "┌─ E2E Detection Report " + "─" * 53]
    lines.append("│  " + "  ".join(h.ljust(w) for h, w in zip(header, col_w)))
    lines.append("│  " + sep)

    prev_section = ""
    for policy_key, section_label in _POLICY_GROUPS:
        group = by_policy_list.get(policy_key, [])
        if not group:
            continue
        if section_label != prev_section:
            lines.append("│")
            lines.append(f"│  ── {section_label}")
            prev_section = section_label
        for r in group:
            row = [str(r.seed), r.task_kind, r.policy, str(r.steps), r.outcome, r.detection]
            lines.append("│  " + "  ".join(v.ljust(w) for v, w in zip(row, col_w)))

    lines.append("│  " + sep)

    correct  = [r for r in results if r.detection == "CORRECT"]
    wrong    = [r for r in results if r.detection == "WRONG"]
    accuracy = len(correct) / len(results) * 100 if results else 0.0

    lines.append("│")
    lines.append(f"│  Total episodes : {len(results)}")
    lines.append(f"│  CORRECT        : {len(correct)}")
    lines.append(f"│  WRONG          : {len(wrong)}")
    lines.append(f"│  Accuracy       : {accuracy:.0f}%")
    lines.append("│")
    lines.append("└" + "─" * 77)

    print("\n".join(lines))

    for r in wrong:
        if r.report:
            print(render_episode_report(r.report))
