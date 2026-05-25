from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Literal, Union

from gymnasium import spaces

from gym.ui_actions import (
    CancelOrder,
    CategoryFilter,
    FilterProducts,
    FormField,
    GoTo,
    OpenProduct,
    SortOption,
    StorePage,
    UiButton,
    UiClick,
    UiFill,
    ViewOrder,
    resolve_ui_action,
)


class ActionKind(IntEnum):
    CLICK = 0
    TYPE = 1
    SCROLL = 2
    NAVIGATE = 3


@dataclass(frozen=True)
class Click:
    selector: str
    type: Literal["click"] = "click"


@dataclass(frozen=True)
class TypeText:
    selector: str
    text: str
    type: Literal["type"] = "type"


@dataclass(frozen=True)
class Scroll:
    delta_y: float
    type: Literal["scroll"] = "scroll"


@dataclass(frozen=True)
class Navigate:
    url: str
    type: Literal["navigate"] = "navigate"


Action = Union[Click, TypeText, Scroll, Navigate]


def click(selector: str) -> Click:
    return Click(selector=selector)


def type_text(selector: str, text: str) -> TypeText:
    return TypeText(selector=selector, text=text)


def scroll(delta_y: float) -> Scroll:
    return Scroll(delta_y=delta_y)


def navigate(url: str) -> Navigate:
    return Navigate(url=url)


def make_action_space() -> spaces.OneOf:
    """Gymnasium action space: OneOf index selects click, type, scroll, or navigate."""
    return spaces.OneOf(
        (
            spaces.Dict({"selector": spaces.Text(max_length=512)}),
            spaces.Dict(
                {
                    "selector": spaces.Text(max_length=512),
                    "text": spaces.Text(max_length=2_048),
                }
            ),
            spaces.Dict(
                {"delta_y": spaces.Box(low=-10_000, high=10_000, shape=(), dtype=float)}
            ),
            spaces.Dict({"url": spaces.Text(max_length=2_048)}),
        )
    )


def action_to_dict(action: Action) -> dict[str, Any]:
    return asdict(action)


def action_from_oneof(sample: tuple[int, dict[str, Any]]) -> Action:
    index, payload = sample
    if index == ActionKind.CLICK:
        return Click(selector=payload["selector"])
    if index == ActionKind.TYPE:
        return TypeText(selector=payload["selector"], text=payload["text"])
    if index == ActionKind.SCROLL:
        return Scroll(delta_y=float(payload["delta_y"]))
    if index == ActionKind.NAVIGATE:
        return Navigate(url=payload["url"])
    raise ValueError(f"unsupported action space index: {index}")


def action_to_oneof(action: Action) -> tuple[int, dict[str, Any]]:
    if isinstance(action, Click):
        return ActionKind.CLICK, {"selector": action.selector}
    if isinstance(action, TypeText):
        return ActionKind.TYPE, {"selector": action.selector, "text": action.text}
    if isinstance(action, Scroll):
        return ActionKind.SCROLL, {"delta_y": action.delta_y}
    if isinstance(action, Navigate):
        return ActionKind.NAVIGATE, {"url": action.url}
    raise TypeError(f"unsupported action type: {type(action)!r}")


def coerce_ui_action(action: dict[str, Any]):
    action_type = action["type"]
    if action_type == "goto":
        return GoTo(page=StorePage(action["page"]))
    if action_type == "filter_products":
        return FilterProducts(
            category=CategoryFilter(action.get("category", "")),
            sort=SortOption(action.get("sort", SortOption.ALPHA.value)),
        )
    if action_type == "open_product":
        return OpenProduct(sku=action["sku"])
    if action_type == "ui_click":
        return UiClick(button=UiButton(action["button"]))
    if action_type == "ui_fill":
        return UiFill(
            field=FormField(action["field"]),
            value=str(action["value"]),
            sku=action.get("sku"),
        )
    if action_type == "view_order":
        return ViewOrder(order_id=int(action["order_id"]))
    if action_type == "cancel_order":
        return CancelOrder(order_id=int(action["order_id"]))
    raise ValueError(f"unsupported ui action type: {action_type}")


def coerce_action(action: Action | tuple[int, dict[str, Any]] | dict[str, Any]) -> Action:
    if isinstance(action, (Click, TypeText, Scroll, Navigate)):
        return action
    if isinstance(action, (GoTo, FilterProducts, OpenProduct, UiClick, UiFill, ViewOrder, CancelOrder)):
        return coerce_action(resolve_ui_action(action))
    if isinstance(action, tuple):
        return action_from_oneof(action)
    if isinstance(action, dict):
        action_type = action.get("type")
        if action_type in {
            "goto",
            "filter_products",
            "open_product",
            "ui_click",
            "ui_fill",
            "view_order",
            "cancel_order",
        }:
            return coerce_action(resolve_ui_action(coerce_ui_action(action)))
        if action_type == "click":
            return Click(selector=action["selector"])
        if action_type == "type":
            return TypeText(selector=action["selector"], text=action.get("text", ""))
        if action_type == "scroll":
            return Scroll(delta_y=float(action.get("delta_y", 500)))
        if action_type == "navigate":
            return Navigate(url=action["url"])
        raise ValueError(f"unsupported action type: {action_type}")
    raise TypeError(f"unsupported action value: {action!r}")
