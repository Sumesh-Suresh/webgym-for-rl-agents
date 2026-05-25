from __future__ import annotations

import random
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from playwright.sync_api import Page

from app import db


def _max_order_id(db_path: str | Path) -> int:
    with db.connect(db_path) as conn:
        row = conn.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM orders").fetchone()
    return int(row["max_id"])


def _order_line_items(db_path: str | Path, order_id: int) -> list[sqlite3.Row]:
    return db.get_order_items(db_path, order_id)


def _discounted_total(subtotal_cents: int, percent_off: int) -> int:
    return subtotal_cents - subtotal_cents * percent_off // 100


def _new_placed_orders(db_path: str | Path, baseline_order_id: int) -> list[sqlite3.Row]:
    with db.connect(db_path) as conn:
        return conn.execute(
            """
            SELECT *
            FROM orders
            WHERE id > ? AND status = 'placed'
            ORDER BY id
            """,
            (baseline_order_id,),
        ).fetchall()


class AbstractEcommerceTask(ABC):
    """BrowserGym-style task: sampled once per episode after the DB is seeded."""

    kind: ClassVar[str]

    def __init__(self, seed: int, db_path: Path, rng: random.Random) -> None:
        self.seed = seed
        self.db_path = db_path
        self.rng = rng
        self.baseline_order_id = _max_order_id(db_path)

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable slug used for observation artifact paths."""

    @property
    @abstractmethod
    def instruction(self) -> str:
        """Natural-language goal shown to the agent after reset."""

    @abstractmethod
    def check_success(self, db_path: str | Path) -> bool:
        """Return whether backend state satisfies this episode's goal."""

    def validate(
        self, page: Page, db_path: str | Path
    ) -> tuple[float, bool, str, dict[str, Any]]:
        """BrowserGym validate API: reward, done, user_message, task_info."""
        success = self.check_success(db_path)
        return float(success), success, "", {"success": success}

    @classmethod
    def sample(cls, seed: int, db_path: Path) -> AbstractEcommerceTask:
        rng = random.Random(seed)
        task_cls = rng.choice(_TASK_CLASSES)
        return task_cls(seed, db_path, rng)


class BuyCheapestInCategoryTask(AbstractEcommerceTask):
    kind = "buy_cheapest_in_category"

    def __init__(self, seed: int, db_path: Path, rng: random.Random) -> None:
        super().__init__(seed, db_path, rng)
        with db.connect(db_path) as conn:
            categories = [
                row["category"]
                for row in conn.execute(
                    "SELECT DISTINCT category FROM products ORDER BY category"
                ).fetchall()
            ]
        self.category = rng.choice(categories)
        self.shipping_address = rng.choice(db.ADDRESS_LIST)
        self.expected_sku = db.cheapest_product_in_category(db_path, self.category)["sku"]

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.category.lower()}"

    @property
    def instruction(self) -> str:
        return (
            f"Buy the cheapest item in the '{self.category}' category and ship it to "
            f"{self.shipping_address}."
        )

    def check_success(self, db_path: str | Path) -> bool:
        for order in _new_placed_orders(db_path, self.baseline_order_id):
            if order["shipping_address"] != self.shipping_address:
                continue
            items = _order_line_items(db_path, order["id"])
            if len(items) == 1 and items[0]["sku"] == self.expected_sku and items[0]["quantity"] == 1:
                return True
        return False


class ApplyCouponWithQuantityTask(AbstractEcommerceTask):
    kind = "apply_coupon_with_quantity"

    def __init__(self, seed: int, db_path: Path, rng: random.Random) -> None:
        super().__init__(seed, db_path, rng)
        with db.connect(db_path) as conn:
            products = conn.execute("SELECT sku, price_cents FROM products ORDER BY sku").fetchall()
        product = rng.choice(products)
        self.sku = product["sku"]
        self.unit_price_cents = int(product["price_cents"])
        self.quantity = rng.randint(1, 3)
        self.coupon_code = rng.choice(sorted(db.COUPON_CODE_MAP))
        self.percent_off = db.COUPON_CODE_MAP[self.coupon_code]
        subtotal = self.unit_price_cents * self.quantity
        self.expected_total = _discounted_total(subtotal, self.percent_off)

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.sku.lower()}"

    @property
    def instruction(self) -> str:
        return (
            f"Add {self.quantity} unit(s) of {self.sku} to the cart, apply coupon "
            f"{self.coupon_code}, and complete checkout."
        )

    def check_success(self, db_path: str | Path) -> bool:
        for order in _new_placed_orders(db_path, self.baseline_order_id):
            if order["coupon_code"] != self.coupon_code or order["total_cents"] != self.expected_total:
                continue
            items = _order_line_items(db_path, order["id"])
            if (
                len(items) == 1
                and items[0]["sku"] == self.sku
                and items[0]["quantity"] == self.quantity
            ):
                return True
        return False


class CancelRecentOrderTask(AbstractEcommerceTask):
    kind = "cancel_recent_order"

    @property
    def name(self) -> str:
        return self.kind

    @property
    def instruction(self) -> str:
        return "Cancel the most recent existing order in the account."

    def check_success(self, db_path: str | Path) -> bool:
        orders = db.list_orders(db_path)
        if not orders:
            return False
        return orders[0]["status"] == "cancelled"


_TASK_CLASSES: tuple[type[AbstractEcommerceTask], ...] = (
    BuyCheapestInCategoryTask,
    ApplyCouponWithQuantityTask,
    CancelRecentOrderTask,
)
