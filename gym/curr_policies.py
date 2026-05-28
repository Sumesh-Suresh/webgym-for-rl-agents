import random
from typing import Any
from urllib.parse import quote

from gym.env import EcommerceEnv
from gym.task import (
    ApplyCouponWithQuantityTask,
    BuyCheapestInCategoryTask,
    CancelRecentOrderTask,
)

ActionSpaceFormat = tuple[int, dict[str, Any]]

CLICK = 0
TYPE = 1
SCROLL = 2
NAVIGATE = 3


def click(selector: str) -> ActionSpaceFormat:
    return CLICK, {"selector": selector}


def type(selector: str, text: str | int) -> ActionSpaceFormat:
    return TYPE, {"selector": selector, "text": str(text)}


def scroll(delta_y: float) -> ActionSpaceFormat:
    return SCROLL, {"delta_y": delta_y}


def navigate(url: str) -> ActionSpaceFormat:
    return NAVIGATE, {"url": url}


def scripted_oracle_actions(env: EcommerceEnv) -> list[ActionSpaceFormat]:
    if env.page is None or env.task is None or env.db_path is None:
        raise RuntimeError("oracle requires a reset environment")

    task = env.task

    if isinstance(task, BuyCheapestInCategoryTask):
        category = quote(task.category, safe="")
        sku = quote(task.expected_sku, safe="")
        return [
            navigate(f"/?category={category}&sort=price_asc"),
            navigate(f"/product/{sku}"),
            click("[data-testid='add-to-cart']"),
            click("[data-testid='checkout-link']"),
            type("[data-testid='shipping-address']", task.shipping_address),
            click("[data-testid='place-order']"),
        ]

    if isinstance(task, ApplyCouponWithQuantityTask):
        sku = quote(task.sku, safe="")
        return [
            navigate("/"),
            navigate(f"/product/{sku}"),
            type("[data-testid='quantity']", task.quantity),
            click("[data-testid='add-to-cart']"),
            type("[data-testid='coupon-code']", task.coupon_code),
            click("[data-testid='apply-coupon']"),
            click("[data-testid='checkout-link']"),
            type("[data-testid='shipping-address']", task.shipping_address),
            click("[data-testid='place-order']"),
        ]

    if isinstance(task, CancelRecentOrderTask):
        return [
            navigate("/orders"),
            click(f"[data-testid='cancel-order-{task.target_order_id}']"),
        ]

    raise ValueError(f"no oracle for task kind: {task.kind}")


def random_action(rng: random.Random) -> ActionSpaceFormat:
    choices: list[ActionSpaceFormat] = [
        navigate("/"),
        navigate("/cart"),
        navigate("/orders"),
        scroll(rng.choice([-500, 500])),
    ]
    return rng.choice(choices)