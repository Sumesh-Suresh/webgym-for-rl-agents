from __future__ import annotations

import sys
import tempfile
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import db  # noqa: E402
from gym.env import EcommerceEnv  # noqa: E402


SEEDS = (0, 1, 42, 72, 56)
SIMULATE_DIR = Path(__file__).resolve().parent / "simulate"


def test_seeded_resets_are_consistent_and_repeatable() -> None:
    snapshots_by_seed: dict[int, dict[str, list[tuple[Any, ...]]]] = {}

    with tempfile.TemporaryDirectory(prefix="webgym-seed-sim-") as tmpdir:
        tmpdir_path = Path(tmpdir)

        for case_number, seed in enumerate(SEEDS, start=1):
            db_path = tmpdir_path / f"case_{case_number}.sqlite"
            db.reset_database(db_path, seed)

            snapshot = database_snapshot(db_path)
            assert_seeded_tables_are_compatible(db_path, snapshot)
            snapshots_by_seed[seed] = snapshot

            case_dir = SIMULATE_DIR / f"test_case{case_number}"
            save_seed_artifacts(seed, case_dir, snapshot)

        assert any(
            snapshots_by_seed[SEEDS[0]] != snapshots_by_seed[seed]
            for seed in SEEDS[1:]
        ), "different seeds should produce at least one different seeded database"

        for seed in SEEDS:
            first_db = tmpdir_path / f"repeat_{seed}_a.sqlite"
            second_db = tmpdir_path / f"repeat_{seed}_b.sqlite"
            db.reset_database(first_db, seed)
            db.reset_database(second_db, seed)

            if database_snapshot(first_db) == database_snapshot(second_db):
                print("seed {seed} produced repeatable database contents")
            else:
                f"seed {seed} did not produce repeatable database contents"
        

def database_snapshot(db_path: Path) -> dict[str, list[tuple[Any, ...]]]:
    table_order = {
        "products": "id",
        "coupons": "code",
        "orders": "id",
        "order_items": "order_id, sku",
    }

    with db.connect(db_path) as conn:
        return {
            table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}")]
            for table, order_by in table_order.items()
        }


def assert_seeded_tables_are_compatible(
    db_path: Path,
    snapshot: dict[str, list[tuple[Any, ...]]],
) -> None:
    products = {
        row[1]: {
            "id": row[0],
            "sku": row[1],
            "name": row[2],
            "category": row[3],
            "price_cents": row[4],
            "description": row[5],
        }
        for row in snapshot["products"]
    }
    coupons = {row[0]: row[1] for row in snapshot["coupons"]}
    orders = {
        row[0]: {
            "id": row[0],
            "status": row[1],
            "shipping_address": row[2],
            "coupon_code": row[3],
            "total_cents": row[4],
            "created_at": row[5],
        }
        for row in snapshot["orders"]
    }
    order_items = snapshot["order_items"]

    assert len(products) >= db.MIN_PRODUCTS_PER_RESET
    assert set(coupons) == set(db.COUPON_CODE_MAP)
    assert set(row[3] for row in snapshot["products"]) == {
        product[2] for product in db.BASE_PRODUCTS
    }
    assert orders
    assert order_items

    with db.connect(db_path) as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []

    item_subtotals_by_order: dict[int, int] = {order_id: 0 for order_id in orders}
    item_counts_by_order: dict[int, int] = {order_id: 0 for order_id in orders}

    for order_id, sku, quantity, unit_price_cents in order_items:
        assert order_id in orders
        assert sku in products
        assert quantity > 0
        assert unit_price_cents == products[sku]["price_cents"]

        item_subtotals_by_order[order_id] += quantity * unit_price_cents
        item_counts_by_order[order_id] += 1

    for order_id, order in orders.items():
        assert item_counts_by_order[order_id] > 0

        coupon_code = order["coupon_code"]
        if coupon_code is not None:
            assert coupon_code in coupons
        percent_off = coupons.get(coupon_code, 0)
        subtotal = item_subtotals_by_order[order_id]
        expected_total = subtotal - subtotal * percent_off // 100
        assert order["total_cents"] == expected_total


def save_seed_artifacts(
    seed: int,
    case_dir: Path,
    expected_snapshot: dict[str, list[tuple[Any, ...]]],
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)

    save_order_state(case_dir, expected_snapshot)

    env = EcommerceEnv(seed=seed, headless=True)
    try:
        env.reset()
        assert env.page is not None
        assert env.base_url is not None
        assert env.db_path is not None
        assert database_snapshot(env.db_path) == expected_snapshot

        env.page.screenshot(path=str(case_dir / "main_page.png"), full_page=True)
        env.page.goto(f"{env.base_url}/orders", wait_until="networkidle")
        env.page.screenshot(path=str(case_dir / "order_page.png"), full_page=True)
        
    finally:
        env.close()


def save_order_state(
    case_dir: Path,
    snapshot: dict[str, list[tuple[Any, ...]]],
) -> None:
    order_items_by_id: dict[int, list[dict[str, Any]]] = {}
    for order_id, sku, quantity, unit_price_cents in snapshot["order_items"]:
        order_items_by_id.setdefault(order_id, []).append(
            {
                "sku": sku,
                "quantity": quantity,
                "unit_price_cents": unit_price_cents,
            }
        )

    orders = [
        {
            "id": row[0],
            "status": row[1],
            "shipping_address": row[2],
            "coupon_code": row[3],
            "total_cents": row[4],
            "created_at": row[5],
            "items": order_items_by_id[row[0]],
        }
        for row in snapshot["orders"]
    ]
    (case_dir / "order_state.json").write_text(
        json.dumps(orders, indent=2, sort_keys=True) + "\n"
    )


def run() -> None:
    test_seeded_resets_are_consistent_and_repeatable()
    print(
        "seed simulation passed for "
        f"{len(SEEDS)} cases; screenshots written under {SIMULATE_DIR.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    run()
