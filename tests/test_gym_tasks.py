"""DB-level task checks without Playwright."""

from __future__ import annotations

import tempfile
from pathlib import Path

from app import db
from app.main import create_app
from fastapi.testclient import TestClient

from gym.tasks import SNAPSHOTS, TASKS


def _run_policy(task_id: str, client: TestClient, db_path: Path, snapshot: dict) -> None:
    if task_id == "buy_cheapest_in_category":
        cheapest = db.cheapest_product_in_category(db_path, "Electronics")
        client.post("/cart/add", data={"sku": cheapest["sku"], "quantity": "1"})
        client.post(
            "/checkout",
            data={"shipping_address": "123 Main St, Springfield, IL 62701"},
        )
    elif task_id == "apply_coupon_with_quantity":
        client.post("/cart/add", data={"sku": "SKU-E7421", "quantity": "2"})
        client.post("/cart/coupon", data={"coupon_code": "SAVE10"})
        client.post(
            "/checkout",
            data={"shipping_address": "123 Main St, Springfield, IL 62701"},
        )
    elif task_id == "cancel_recent_order":
        target = snapshot["cancel_target_id"]
        client.post(f"/orders/{target}/cancel")
    else:
        raise AssertionError(f"unknown task {task_id!r}")


def test_tasks_verify_against_app() -> None:
    for seed in (0, 1, 42):
        _verify_all_tasks(seed)


def _verify_all_tasks(seed: int) -> None:
    for task_id, task in TASKS.items():
        with tempfile.TemporaryDirectory() as workdir:
            db_path = Path(workdir) / "store.sqlite"
            db.reset_database(db_path, seed)
            if task.seed_extra:
                task.seed_extra(str(db_path), seed)
            snapshot = SNAPSHOTS[task_id](str(db_path))
            client = TestClient(create_app(db_path, seed))
            _run_policy(task_id, client, db_path, snapshot)
            success, info = task.verify(str(db_path), snapshot)
            assert success, info
