"""Task definitions for the e-commerce gym.

A task is a tuple of:
  - id, instruction
  - seed_extra: optional hook that inserts task-specific seed data into the DB
    after the base reset (e.g., a cancelable pre-existing order). All inserts
    are deterministic given the gym seed.
  - verify: reads the DB and returns (success: bool, info: dict). Verifiers
    NEVER read rendered HTML.
  - oracle: a scripted solver that drives a Playwright page to completion.
    Used by the parallel rollout demo and by tests as a ground-truth policy.

Adding a new task means appending one Task here. The env code is unchanged.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable, Optional

from app import db


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

VerifyResult = tuple[bool, dict]
SeedFn = Callable[[str, int], None]                      # (db_path, seed) -> None
VerifyFn = Callable[[str, dict], VerifyResult]            # (db_path, snapshot) -> result
OracleFn = Callable[["Page"], None]                       # noqa: F821  (Playwright Page)


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    seed_extra: Optional[SeedFn]
    verify: VerifyFn
    oracle: OracleFn


# ---------------------------------------------------------------------------
# Helpers shared across tasks
# ---------------------------------------------------------------------------

def _snapshot(db_path: str) -> dict:
    """Capture pre-step DB state so verifiers can detect changes, not just
    final state. Used by tasks where the agent must *create* something new."""
    with db.connect(db_path) as conn:
        order_ids = [row["id"] for row in conn.execute("SELECT id FROM orders").fetchall()]
    return {"order_ids_at_reset": set(order_ids)}


def _orders_created_during_episode(db_path: str, snapshot: dict) -> list[sqlite3.Row]:
    with db.connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM orders ORDER BY id").fetchall()
    return [r for r in rows if r["id"] not in snapshot["order_ids_at_reset"]]


# ---------------------------------------------------------------------------
# Task 1: buy_cheapest_in_category (Electronics → 123 Main St)
# ---------------------------------------------------------------------------

EXPECTED_ADDRESS = "123 Main St, Springfield, IL 62701"


def _verify_buy_cheapest(db_path: str, snapshot: dict) -> VerifyResult:
    new_orders = _orders_created_during_episode(db_path, snapshot)
    info: dict = {"new_order_count": len(new_orders)}
    if len(new_orders) != 1:
        info["reason"] = f"expected exactly 1 new order, got {len(new_orders)}"
        return False, info

    order = new_orders[0]
    info["order_id"] = order["id"]

    if order["status"] != "placed":
        info["reason"] = f"order status is {order['status']!r}, expected 'placed'"
        return False, info

    if order["shipping_address"].strip() != EXPECTED_ADDRESS:
        info["reason"] = f"wrong shipping address: {order['shipping_address']!r}"
        return False, info

    items = db.get_order_items(db_path, order["id"])
    info["items"] = [(r["sku"], r["quantity"]) for r in items]
    if len(items) != 1:
        info["reason"] = f"expected exactly 1 line item, got {len(items)}"
        return False, info

    item = items[0]
    cheapest = db.cheapest_product_in_category(db_path, "Electronics")
    if item["sku"] != cheapest["sku"]:
        info["reason"] = (
            f"bought {item['sku']!r} but cheapest Electronics is {cheapest['sku']!r}"
        )
        return False, info

    if item["quantity"] != 1:
        info["reason"] = f"expected quantity 1, got {item['quantity']}"
        return False, info

    return True, info


def _oracle_buy_cheapest(page) -> None:
    cheapest_sku = _cheapest_electronics_sku_via_dom(page)
    base = page.evaluate("() => window.location.origin")
    page.goto(f"{base}/product/{cheapest_sku}")
    page.locator('[data-testid="quantity"]').fill("1")
    page.locator('[data-testid="add-to-cart"]').click()
    page.locator('[data-testid="checkout-link"]').click()
    page.locator('[data-testid="shipping-address"]').fill(EXPECTED_ADDRESS)
    page.locator('[data-testid="place-order"]').click()


def _cheapest_electronics_sku_via_dom(page) -> str:
    """Drive the filter UI and read off the first row. The oracle is allowed to
    use the DOM for *navigation* — only verifiers are forbidden from doing so."""
    base = page.evaluate("() => window.location.origin")
    page.goto(f"{base}/?category=Electronics&sort=price_asc")
    first_link = page.locator('table[aria-label="Product list"] tbody tr a').first
    href = first_link.get_attribute("href") or ""
    # href is "/product/SKU-EXXXX"
    return href.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Task 2: apply_coupon_with_quantity (2 × SKU-E7421, SAVE10)
# ---------------------------------------------------------------------------

EXPECTED_SKU = "SKU-E7421"
EXPECTED_QTY = 2
EXPECTED_COUPON = "SAVE10"


def _seed_coupon_task(db_path: str, seed: int) -> None:
    db.ensure_products(db_path, [EXPECTED_SKU])


def _verify_coupon_quantity(db_path: str, snapshot: dict) -> VerifyResult:
    new_orders = _orders_created_during_episode(db_path, snapshot)
    info: dict = {"new_order_count": len(new_orders)}
    if len(new_orders) != 1:
        info["reason"] = f"expected exactly 1 new order, got {len(new_orders)}"
        return False, info

    order = new_orders[0]
    info["order_id"] = order["id"]
    info["coupon"] = order["coupon_code"]

    if order["status"] != "placed":
        info["reason"] = f"order status is {order['status']!r}"
        return False, info

    if (order["coupon_code"] or "").upper() != EXPECTED_COUPON:
        info["reason"] = f"expected coupon {EXPECTED_COUPON!r}, got {order['coupon_code']!r}"
        return False, info

    items = db.get_order_items(db_path, order["id"])
    info["items"] = [(r["sku"], r["quantity"]) for r in items]
    if len(items) != 1 or items[0]["sku"] != EXPECTED_SKU:
        info["reason"] = f"expected single line item with sku {EXPECTED_SKU!r}"
        return False, info
    if items[0]["quantity"] != EXPECTED_QTY:
        info["reason"] = f"expected qty {EXPECTED_QTY}, got {items[0]['quantity']}"
        return False, info

    # Independently recompute the expected total so a coupon-skipping agent fails
    # even if it stumbles into the right items.
    product = db.get_product(db_path, EXPECTED_SKU)
    subtotal = product["price_cents"] * EXPECTED_QTY
    expected_total = subtotal - subtotal * 10 // 100
    if order["total_cents"] != expected_total:
        info["reason"] = (
            f"total {order['total_cents']} != expected {expected_total} "
            "(coupon may not have applied)"
        )
        return False, info

    return True, info


def _oracle_coupon_quantity(page) -> None:
    base = page.evaluate("() => window.location.origin")
    page.goto(f"{base}/product/{EXPECTED_SKU}")
    page.locator('[data-testid="quantity"]').fill(str(EXPECTED_QTY))
    page.locator('[data-testid="add-to-cart"]').click()
    page.locator('[data-testid="coupon-code"]').fill(EXPECTED_COUPON)
    page.locator('[data-testid="apply-coupon"]').click()
    page.locator('[data-testid="checkout-link"]').click()
    page.locator('[data-testid="shipping-address"]').fill(EXPECTED_ADDRESS)
    page.locator('[data-testid="place-order"]').click()


# ---------------------------------------------------------------------------
# Task 3: cancel_recent_order (pre-seeded)
# ---------------------------------------------------------------------------

# Fixed timestamps so order IDs and ordering are stable across resets.
_SEED_ORDER_TIMESTAMPS = [
    "2025-01-15T09:00:00Z",
    "2025-02-03T14:30:00Z",
    "2025-03-21T11:45:00Z",  # most recent → the cancelable target
]


def _seed_orders(db_path: str, seed: int) -> None:
    """Insert three deterministic pre-existing orders. The most recent is the
    cancel target. We bypass `db.create_order` because it stamps now()."""
    # Use the same SKUs each time so the orders are fully reproducible.
    pre_seeded_orders = [
        {
            "address": "456 Oak Ave, Madison, WI 53703",
            "items": [("SKU-O1001", 2), ("SKU-O2002", 1)],
        },
        {
            "address": "789 Pine Rd, Boulder, CO 80302",
            "items": [("SKU-H1001", 1)],
        },
        {
            "address": "321 Elm St, Austin, TX 78701",
            "items": [("SKU-F2002", 1), ("SKU-F5005", 2)],
        },
    ]
    required_skus = {sku for order in pre_seeded_orders for sku, _ in order["items"]}
    db.ensure_products(db_path, required_skus)

    with db.connect(db_path) as conn:
        # Replace any orders created by the base reset so IDs and timestamps stay stable.
        conn.execute("DELETE FROM order_items")
        conn.execute("DELETE FROM orders")
        for idx, order in enumerate(pre_seeded_orders, start=1):
            subtotal = 0
            unit_prices = {}
            for sku, qty in order["items"]:
                row = conn.execute(
                    "SELECT price_cents FROM products WHERE sku = ?", (sku,)
                ).fetchone()
                unit_prices[sku] = row["price_cents"]
                subtotal += row["price_cents"] * qty
            conn.execute(
                """
                INSERT INTO orders (id, status, shipping_address, coupon_code, total_cents, created_at)
                VALUES (?, 'placed', ?, NULL, ?, ?)
                """,
                (idx, order["address"], subtotal, _SEED_ORDER_TIMESTAMPS[idx - 1]),
            )
            conn.executemany(
                """
                INSERT INTO order_items (order_id, sku, quantity, unit_price_cents)
                VALUES (?, ?, ?, ?)
                """,
                [(idx, sku, qty, unit_prices[sku]) for sku, qty in order["items"]],
            )


def _most_recent_pre_seeded_order_id(db_path: str) -> int:
    """The verifier needs to know which order was the cancel target *at reset
    time*. Because seeding is deterministic the answer is always the order with
    the latest created_at among the pre-seeded ones."""
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM orders ORDER BY created_at DESC, id DESC LIMIT 1"
        ).fetchone()
    return row["id"]


def _verify_cancel_recent(db_path: str, snapshot: dict) -> VerifyResult:
    info: dict = {}
    target_id = snapshot["cancel_target_id"]
    info["target_order_id"] = target_id

    order = db.get_order(db_path, target_id)
    if order is None:
        info["reason"] = f"target order {target_id} no longer exists"
        return False, info

    info["status"] = order["status"]
    if order["status"] != "cancelled":
        info["reason"] = f"target order status is {order['status']!r}, expected 'cancelled'"
        return False, info

    # Catch agents that cancel multiple orders or place new ones.
    with db.connect(db_path) as conn:
        other_cancelled = conn.execute(
            "SELECT id FROM orders WHERE status = 'cancelled' AND id != ?",
            (target_id,),
        ).fetchall()
    if other_cancelled:
        info["reason"] = f"cancelled other orders too: {[r['id'] for r in other_cancelled]}"
        return False, info

    new_orders = _orders_created_during_episode(db_path, snapshot)
    if new_orders:
        info["reason"] = f"agent created {len(new_orders)} extra order(s)"
        return False, info

    return True, info


def _snapshot_cancel(db_path: str) -> dict:
    snap = _snapshot(db_path)
    snap["cancel_target_id"] = _most_recent_pre_seeded_order_id(db_path)
    return snap


def _oracle_cancel_recent(page) -> None:
    base = page.evaluate("() => window.location.origin")
    page.goto(f"{base}/orders")
    # The orders page lists orders newest-first; the first cancel button is the target.
    page.locator('button[data-oracle-action="cancel-order"]').first.click()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASKS: dict[str, Task] = {
    "buy_cheapest_in_category": Task(
        id="buy_cheapest_in_category",
        instruction=(
            "Buy the cheapest item in the 'Electronics' category and ship it to "
            f"{EXPECTED_ADDRESS}."
        ),
        seed_extra=None,
        verify=_verify_buy_cheapest,
        oracle=_oracle_buy_cheapest,
    ),
    "apply_coupon_with_quantity": Task(
        id="apply_coupon_with_quantity",
        instruction=(
            f"Add {EXPECTED_QTY} units of {EXPECTED_SKU} to the cart, apply coupon "
            f"{EXPECTED_COUPON}, and complete checkout."
        ),
        seed_extra=_seed_coupon_task,
        verify=_verify_coupon_quantity,
        oracle=_oracle_coupon_quantity,
    ),
    "cancel_recent_order": Task(
        id="cancel_recent_order",
        instruction="Cancel the most recent existing order in the account.",
        seed_extra=_seed_orders,
        verify=_verify_cancel_recent,
        oracle=_oracle_cancel_recent,
    ),
}


# Per-task snapshot factory: cancel needs to remember the target ID; the others
# just need the set of pre-existing order IDs.
SNAPSHOTS: dict[str, Callable[[str], dict]] = {
    "buy_cheapest_in_category": _snapshot,
    "apply_coupon_with_quantity": _snapshot,
    "cancel_recent_order": _snapshot_cancel,
}


def get_task(task_id: str) -> Task:
    if task_id not in TASKS:
        raise KeyError(f"unknown task {task_id!r}. Available: {sorted(TASKS)}")
    return TASKS[task_id]
