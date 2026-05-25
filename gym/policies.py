from __future__ import annotations

import random

from app import db
from gym.actions import Action, scroll
from gym.env import EcommerceEnv
from gym.task import (
    ApplyCouponWithQuantityTask,
    BuyCheapestInCategoryTask,
    CancelRecentOrderTask,
)
from gym.ui_actions import (
    CategoryFilter,
    FormField,
    SortOption,
    StorePage,
    UiAction,
    UiButton,
    cancel_order,
    click_ui,
    fill_field,
    filter_products,
    goto,
    open_product,
    view_order,
)


def _newest_placed_order_id(db_path) -> int:
    for order in db.list_orders(db_path):
        if order["status"] == "placed":
            return int(order["id"])
    raise RuntimeError("no placed order found")


def _newest_order_id(db_path) -> int:
    orders = db.list_orders(db_path)
    if not orders:
        raise RuntimeError("no orders found")
    return int(orders[0]["id"])


def scripted_oracle_actions(env: EcommerceEnv) -> list[UiAction | Action]:
    if env.page is None or env.task is None or env.db_path is None:
        raise RuntimeError("oracle requires a reset environment")

    task = env.task
    db_path = env.db_path

    if isinstance(task, BuyCheapestInCategoryTask):
        return [
            filter_products(
                category=CategoryFilter(task.category),
                sort=SortOption.PRICE_ASC,
            ),
            open_product(task.expected_sku),
            click_ui(UiButton.ADD_TO_CART),
            click_ui(UiButton.CHECKOUT_LINK),
            fill_field(FormField.SHIPPING_ADDRESS, task.shipping_address),
            click_ui(UiButton.PLACE_ORDER),
            goto(StorePage.ORDERS),
            view_order(_newest_order_id(db_path)),
        ]

    if isinstance(task, ApplyCouponWithQuantityTask):
        shipping_address = task.rng.choice(db.ADDRESS_LIST)
        return [
            goto(StorePage.HOME),
            open_product(task.sku),
            fill_field(FormField.QUANTITY, task.quantity),
            click_ui(UiButton.ADD_TO_CART),
            fill_field(FormField.COUPON_CODE, task.coupon_code),
            click_ui(UiButton.APPLY_COUPON),
            click_ui(UiButton.CHECKOUT_LINK),
            fill_field(FormField.SHIPPING_ADDRESS, shipping_address),
            click_ui(UiButton.PLACE_ORDER),
            goto(StorePage.ORDERS),
            view_order(_newest_order_id(db_path)),
        ]

    if isinstance(task, CancelRecentOrderTask):
        order_id = _newest_placed_order_id(db_path)
        return [
            goto(StorePage.ORDERS),
            cancel_order(order_id),
            view_order(order_id),
        ]

    raise ValueError(f"no oracle for task kind: {task.kind}")


def random_action(rng: random.Random) -> UiAction | Action:
    choices: list[Action] = [
        goto(StorePage.HOME),
        goto(StorePage.CART),
        goto(StorePage.ORDERS),
        scroll(rng.choice([-500, 500])),
    ]
    return rng.choice(choices)
