"""Tests for task sampling, initialization, and validation — no browser needed."""
from __future__ import annotations

import random

import pytest

from app import db
from gym.task import (
    AbstractEcommerceTask,
    ApplyCouponWithQuantityTask,
    BuyCheapestInCategoryTask,
    CancelRecentOrderTask,
)


# ── Sampling determinism ──────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [1, 7, 42, 100])
def test_task_sampling_is_deterministic(seed, tmp_path):
    p1, p2 = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
    db.reset_database(p1, seed)
    db.reset_database(p2, seed)
    t1 = AbstractEcommerceTask.sample(seed, p1)
    t2 = AbstractEcommerceTask.sample(seed, p2)
    assert type(t1) is type(t2)
    assert t1.instruction == t2.instruction


def test_all_three_task_kinds_reachable(tmp_path):
    """All three task kinds must appear somewhere in the first 200 seeds."""
    path = tmp_path / "s.sqlite"
    kinds_seen: set[str] = set()
    for seed in range(200):
        db.reset_database(path, seed)
        task = AbstractEcommerceTask.sample(seed, path)
        kinds_seen.add(task.kind)
        if len(kinds_seen) == 3:
            break
    assert kinds_seen == {
        "buy_cheapest_in_category",
        "apply_coupon_with_quantity",
        "cancel_recent_order",
    }, f"only saw: {kinds_seen}"


# ── Cancel task sampling guard ────────────────────────────────────────────────

def test_cancel_task_never_sampled_without_placed_order(tmp_path):
    """sample() must not return CancelRecentOrderTask when no placed order exists."""
    path = tmp_path / "s.sqlite"
    for seed in range(200):
        db.reset_database(path, seed)
        has_placed = any(o["status"] == "placed" for o in db.list_orders(path))
        if not has_placed:
            task = AbstractEcommerceTask.sample(seed, path)
            assert not isinstance(task, CancelRecentOrderTask), (
                f"seed={seed}: CancelRecentOrderTask sampled with no placed orders"
            )


# ── BuyCheapestInCategoryTask ─────────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(10))
def test_buy_cheapest_expected_sku_is_actually_cheapest(seed, tmp_db):
    path = tmp_db(seed)
    task = BuyCheapestInCategoryTask(seed, path, random.Random(seed))
    cheapest = db.cheapest_product_in_category(path, task.category)
    assert task.expected_sku == cheapest["sku"]


def test_buy_cheapest_check_success_false_before_order(tmp_db):
    path = tmp_db(1)
    task = BuyCheapestInCategoryTask(1, path, random.Random(1))
    assert not task.check_success(path)


def test_buy_cheapest_check_success_wrong_sku_fails(tmp_db):
    path = tmp_db(1)
    task = BuyCheapestInCategoryTask(1, path, random.Random(1))
    wrong_sku = next(r["sku"] for r in db.list_products(path) if r["sku"] != task.expected_sku)
    db.create_order(path, {wrong_sku: 1}, task.shipping_address, None)
    assert not task.check_success(path)


def test_buy_cheapest_check_success_wrong_address_fails(tmp_db):
    path = tmp_db(1)
    task = BuyCheapestInCategoryTask(1, path, random.Random(1))
    db.create_order(path, {task.expected_sku: 1}, "999 Wrong Ave, Nowhere, ZZ 00000", None)
    assert not task.check_success(path)


def test_buy_cheapest_check_success_correct_order_passes(tmp_db):
    path = tmp_db(1)
    task = BuyCheapestInCategoryTask(1, path, random.Random(1))
    db.create_order(path, {task.expected_sku: 1}, task.shipping_address, None)
    assert task.check_success(path)


# ── ApplyCouponWithQuantityTask ───────────────────────────────────────────────

def test_apply_coupon_check_success_false_before_order(tmp_db):
    path = tmp_db(7)
    task = ApplyCouponWithQuantityTask(7, path, random.Random(7))
    assert not task.check_success(path)


def test_apply_coupon_check_wrong_quantity_fails(tmp_db):
    path = tmp_db(7)
    task = ApplyCouponWithQuantityTask(7, path, random.Random(7))
    db.create_order(path, {task.sku: task.quantity + 1}, task.shipping_address, task.coupon_code)
    assert not task.check_success(path)


def test_apply_coupon_check_missing_coupon_fails(tmp_db):
    path = tmp_db(7)
    task = ApplyCouponWithQuantityTask(7, path, random.Random(7))
    db.create_order(path, {task.sku: task.quantity}, task.shipping_address, None)
    assert not task.check_success(path)


def test_apply_coupon_check_correct_passes(tmp_db):
    path = tmp_db(7)
    task = ApplyCouponWithQuantityTask(7, path, random.Random(7))
    db.create_order(path, {task.sku: task.quantity}, task.shipping_address, task.coupon_code)
    assert task.check_success(path)


def test_apply_coupon_expected_total_matches_discount(tmp_db):
    path = tmp_db(7)
    task = ApplyCouponWithQuantityTask(7, path, random.Random(7))
    order_id = db.create_order(path, {task.sku: task.quantity}, task.shipping_address, task.coupon_code)
    order = db.get_order(path, order_id)
    assert order["total_cents"] == task.expected_total


# ── CancelRecentOrderTask ─────────────────────────────────────────────────────

def _find_seed_with_placed_order(tmp_path) -> tuple[int, object]:
    """Return (seed, db_path) for the first seed that has a placed order."""
    for seed in range(200):
        path = tmp_path / f"s_{seed}.sqlite"
        db.reset_database(path, seed)
        if any(o["status"] == "placed" for o in db.list_orders(path)):
            return seed, path
    raise RuntimeError("no seed in 0..200 produced a placed order")


def test_cancel_task_stores_initial_status(tmp_path):
    seed, path = _find_seed_with_placed_order(tmp_path)
    task = CancelRecentOrderTask(seed, path, random.Random(seed))
    assert hasattr(task, "initial_status")
    assert task.initial_status == "placed"


def test_cancel_task_check_success_false_before_cancel(tmp_path):
    seed, path = _find_seed_with_placed_order(tmp_path)
    task = CancelRecentOrderTask(seed, path, random.Random(seed))
    assert not task.check_success(path)


def test_cancel_task_check_success_true_after_cancel(tmp_path):
    seed, path = _find_seed_with_placed_order(tmp_path)
    task = CancelRecentOrderTask(seed, path, random.Random(seed))
    db.cancel_order(path, task.target_order_id)
    assert task.check_success(path)


def test_cancel_task_targets_most_recent_placed_order(tmp_path):
    """target_order_id must be the most-recent placed order (list_orders sorts by created_at DESC)."""
    seed, path = _find_seed_with_placed_order(tmp_path)
    task = CancelRecentOrderTask(seed, path, random.Random(seed))
    first_placed = next(o for o in db.list_orders(path) if o["status"] == "placed")
    assert task.target_order_id == int(first_placed["id"])


def test_cancel_task_immune_to_new_placed_orders(tmp_path):
    """Placing a newer order after reset must not shift the target or satisfy the verifier."""
    seed, path = _find_seed_with_placed_order(tmp_path)
    task = CancelRecentOrderTask(seed, path, random.Random(seed))
    original_target = task.target_order_id

    # Place a brand-new order (more recent than all seeded orders)
    some_sku = db.list_products(path)[0]["sku"]
    db.create_order(path, {some_sku: 1}, "1 Main St, City, IL 60000", None)

    assert task.target_order_id == original_target, "target_order_id must not shift"

    # Cancelling the new (non-target) order must not satisfy the task
    new_order_id = int(db.list_orders(path)[0]["id"])
    assert new_order_id != original_target
    db.cancel_order(path, new_order_id)
    assert not task.check_success(path)


def test_cancel_task_raises_without_placed_order(tmp_path):
    """Constructing CancelRecentOrderTask directly with no placed order should raise."""
    path = tmp_path / "empty.sqlite"
    # Find a seed with zero orders
    for seed in range(200):
        db.reset_database(path, seed)
        orders = db.list_orders(path)
        if not any(o["status"] == "placed" for o in orders):
            with pytest.raises(ValueError):
                CancelRecentOrderTask(seed, path, random.Random(seed))
            return
    pytest.skip("no seed with zero placed orders found in 0..200")
