from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Union
from urllib.parse import quote, urlparse

from app.main import CATEGORIES, SORT_OPTIONS


class StorePage(str, Enum):
    HOME = "home"
    CART = "cart"
    CHECKOUT = "checkout"
    ORDERS = "orders"


class SortOption(str, Enum):
    ALPHA = "alpha"
    PRICE_ASC = "price_asc"
    PRICE_DESC = "price_desc"


class CategoryFilter(str, Enum):
    ALL = ""
    ELECTRONICS = "Electronics"
    FITNESS = "Fitness"
    HOME = "Home"
    OFFICE = "Office"


class UiButton(str, Enum):
    ADD_TO_CART = "add_to_cart"
    APPLY_COUPON = "apply_coupon"
    CHECKOUT_LINK = "checkout_link"
    PLACE_ORDER = "place_order"
    APPLY_FILTERS = "apply_filters"


class FormField(str, Enum):
    QUANTITY = "quantity"
    SHIPPING_ADDRESS = "shipping_address"
    COUPON_CODE = "coupon_code"
    CART_QUANTITY = "cart_quantity"


_BUTTON_SELECTORS: dict[UiButton, str] = {
    UiButton.ADD_TO_CART: "[data-testid='add-to-cart']",
    UiButton.APPLY_COUPON: "[data-testid='apply-coupon']",
    UiButton.CHECKOUT_LINK: "[data-testid='checkout-link']",
    UiButton.PLACE_ORDER: "[data-testid='place-order']",
    UiButton.APPLY_FILTERS: "[data-testid='apply-filters']",
}

_FIELD_SELECTORS: dict[FormField, str] = {
    FormField.QUANTITY: "[data-testid='quantity']",
    FormField.SHIPPING_ADDRESS: "[data-testid='shipping-address']",
    FormField.COUPON_CODE: "[data-testid='coupon-code']",
}


@dataclass(frozen=True)
class GoTo:
    page: StorePage
    type: Literal["goto"] = "goto"


@dataclass(frozen=True)
class FilterProducts:
    category: CategoryFilter = CategoryFilter.ALL
    sort: SortOption = SortOption.ALPHA
    type: Literal["filter_products"] = "filter_products"


@dataclass(frozen=True)
class OpenProduct:
    sku: str
    type: Literal["open_product"] = "open_product"


@dataclass(frozen=True)
class UiClick:
    button: UiButton
    type: Literal["ui_click"] = "ui_click"


@dataclass(frozen=True)
class UiFill:
    field: FormField
    value: str
    sku: str | None = None
    type: Literal["ui_fill"] = "ui_fill"


@dataclass(frozen=True)
class ViewOrder:
    order_id: int
    type: Literal["view_order"] = "view_order"


@dataclass(frozen=True)
class CancelOrder:
    order_id: int
    type: Literal["cancel_order"] = "cancel_order"


UiAction = Union[GoTo, FilterProducts, OpenProduct, UiClick, UiFill, ViewOrder, CancelOrder]


def goto(page: StorePage) -> GoTo:
    return GoTo(page=page)


def filter_products(
    *,
    category: CategoryFilter | str | None = CategoryFilter.ALL,
    sort: SortOption | str = SortOption.ALPHA,
) -> FilterProducts:
    resolved_category = _coerce_category(category)
    resolved_sort = _coerce_sort(sort)
    return FilterProducts(category=resolved_category, sort=resolved_sort)


def open_product(sku: str) -> OpenProduct:
    return OpenProduct(sku=sku)


def click_ui(button: UiButton) -> UiClick:
    return UiClick(button=button)


def fill_field(field: FormField, value: str | int, *, sku: str | None = None) -> UiFill:
    return UiFill(field=field, value=str(value), sku=sku)


def view_order(order_id: int) -> ViewOrder:
    return ViewOrder(order_id=order_id)


def cancel_order(order_id: int) -> CancelOrder:
    return CancelOrder(order_id=order_id)


def resolve_ui_action(action: UiAction) -> dict[str, Any]:
    if isinstance(action, GoTo):
        return {"type": "navigate", "url": _store_page_path(action.page)}
    if isinstance(action, FilterProducts):
        return {"type": "navigate", "url": _filter_url(action.category, action.sort)}
    if isinstance(action, OpenProduct):
        return {"type": "navigate", "url": f"/product/{quote(action.sku, safe='')}"}
    if isinstance(action, UiClick):
        return {"type": "click", "selector": _BUTTON_SELECTORS[action.button]}
    if isinstance(action, UiFill):
        selector = _field_selector(action.field, action.sku)
        return {"type": "type", "selector": selector, "text": action.value}
    if isinstance(action, ViewOrder):
        return {"type": "click", "selector": f"[data-testid='view-order-{action.order_id}']"}
    if isinstance(action, CancelOrder):
        return {"type": "click", "selector": f"[data-testid='cancel-order-{action.order_id}']"}
    raise TypeError(f"unsupported ui action: {action!r}")


def detect_store_page(url: str) -> StorePage | None:
    path = urlparse(url).path.rstrip("/") or "/"
    if path == "/":
        return StorePage.HOME
    if path == "/cart":
        return StorePage.CART
    if path == "/checkout":
        return StorePage.CHECKOUT
    if path == "/orders":
        return StorePage.ORDERS
    return None


def available_actions(url: str) -> dict[str, list[str]]:
    """Return discriminative action options for the current storefront page."""
    page = detect_store_page(url)
    options: dict[str, list[str]] = {
        "pages": [member.value for member in StorePage],
        "categories": [member.value for member in CategoryFilter],
        "sorts": [member.value for member in SortOption],
    }
    if page is StorePage.HOME:
        options["actions"] = ["filter_products", "open_product", "goto"]
    elif page is StorePage.CART:
        options["actions"] = [
            "ui_fill:cart_quantity",   # update quantity for a specific SKU
            "ui_fill:coupon_code",
            "ui_click:apply_coupon",
            "ui_click:checkout_link",
            "goto",
        ]
    elif page is None and "/product/" in urlparse(url).path:
        options["actions"] = ["ui_fill:quantity", "ui_click:add_to_cart", "goto"]
    elif page is StorePage.CHECKOUT:
        options["actions"] = ["ui_fill:shipping_address", "ui_click:place_order", "goto"]
    elif page is StorePage.ORDERS:
        options["actions"] = ["view_order", "cancel_order", "goto"]
    else:
        options["actions"] = ["goto"]
    return options


def _store_page_path(page: StorePage) -> str:
    if page is StorePage.HOME:
        return "/"
    return f"/{page.value}"


def _filter_url(category: CategoryFilter, sort: SortOption) -> str:
    params: list[str] = []
    if category is not CategoryFilter.ALL:
        params.append(f"category={quote(category.value, safe='')}")
    params.append(f"sort={sort.value}")
    return "/?" + "&".join(params)


def _field_selector(field: FormField, sku: str | None) -> str:
    if field is FormField.CART_QUANTITY:
        if sku is None:
            raise ValueError("cart_quantity fill requires sku")
        return f"[data-testid='cart-quantity-{sku}']"
    return _FIELD_SELECTORS[field]


def _coerce_category(category: CategoryFilter | str | None) -> CategoryFilter:
    if category is None or category == "":
        return CategoryFilter.ALL
    if isinstance(category, CategoryFilter):
        return category
    if category not in CATEGORIES:
        raise ValueError(f"unsupported category: {category}")
    return CategoryFilter(category)


def _coerce_sort(sort: SortOption | str) -> SortOption:
    if isinstance(sort, SortOption):
        return sort
    if sort not in SORT_OPTIONS:
        raise ValueError(f"unsupported sort: {sort}")
    return SortOption(sort)
