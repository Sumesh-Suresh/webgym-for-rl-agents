"""Tests for database seeding: determinism, structural guarantees, FK integrity."""
from __future__ import annotations

import pytest

from app import db


# ── Determinism ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 42, 99, 1000])
def test_same_seed_same_products(seed, tmp_path):
    p1, p2 = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
    db.reset_database(p1, seed)
    db.reset_database(p2, seed)
    snap = lambda p: [(r["sku"], r["category"], r["price_cents"]) for r in db.list_products(p)]
    assert snap(p1) == snap(p2)


@pytest.mark.parametrize("seed", [0, 1, 42, 99, 1000])
def test_same_seed_same_orders(seed, tmp_path):
    p1, p2 = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
    db.reset_database(p1, seed)
    db.reset_database(p2, seed)
    snap = lambda p: [(r["id"], r["status"], r["total_cents"]) for r in db.list_orders(p)]
    assert snap(p1) == snap(p2)


def test_different_seeds_differ(tmp_path):
    p1, p2 = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
    db.reset_database(p1, 1)
    db.reset_database(p2, 999)
    assert [r["sku"] for r in db.list_products(p1)] != [r["sku"] for r in db.list_products(p2)]


# ── Product guarantees ────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(15))
def test_at_least_10_products(seed, tmp_db):
    assert len(db.list_products(tmp_db(seed))) >= db.MIN_PRODUCTS_PER_RESET


@pytest.mark.parametrize("seed", range(15))
def test_at_least_2_products_per_category(seed, tmp_db):
    path = tmp_db(seed)
    by_cat: dict[str, int] = {}
    for p in db.list_products(path):
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
    for cat, count in by_cat.items():
        assert count >= db.MIN_PER_CATEGORY, (
            f"seed={seed}: {cat} has only {count} product(s), need >= {db.MIN_PER_CATEGORY}"
        )


def test_coupons_always_seeded(tmp_db):
    path = tmp_db(1)
    for code, pct in db.COUPON_CODE_MAP.items():
        row = db.get_coupon(path, code)
        assert row is not None, f"coupon {code!r} missing"
        assert int(row["percent_off"]) == pct


# ── Order guarantees ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(20))
def test_order_count_in_range(seed, tmp_db):
    n = len(db.list_orders(tmp_db(seed)))
    assert db.MIN_ORDERS_PER_RESET <= n <= db.MAX_ORDERS_PER_RESET


def test_zero_orders_is_a_valid_state(tmp_path):
    """Some seeds must produce zero orders — a state with no orders is valid."""
    path = tmp_path / "s.sqlite"
    found_zero = False
    for seed in range(200):
        db.reset_database(path, seed)
        if len(db.list_orders(path)) == 0:
            found_zero = True
            break
    assert found_zero, "no seed in 0..200 produced zero orders"


def test_no_forced_placed_order(tmp_path):
    """Orders must not be unconditionally forced to placed — all-cancelled is valid."""
    path = tmp_path / "s.sqlite"
    found_all_cancelled = False
    for seed in range(200):
        db.reset_database(path, seed)
        orders = db.list_orders(path)
        if orders and all(o["status"] == "cancelled" for o in orders):
            found_all_cancelled = True
            break
    assert found_all_cancelled, "no seed in 0..200 had orders that were all cancelled"


# ── FK integrity ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(10))
def test_fk_integrity(seed, tmp_db):
    path = tmp_db(seed)
    with db.connect(path) as conn:
        orphan_items = conn.execute(
            "SELECT oi.order_id FROM order_items oi "
            "LEFT JOIN orders o ON oi.order_id = o.id WHERE o.id IS NULL"
        ).fetchall()
        assert not orphan_items, f"orphan order_items: {list(orphan_items)}"

        unknown_skus = conn.execute(
            "SELECT oi.sku FROM order_items oi "
            "LEFT JOIN products p ON oi.sku = p.sku WHERE p.sku IS NULL"
        ).fetchall()
        assert not unknown_skus, f"order_items with unknown skus: {list(unknown_skus)}"
