"""Tests for action coercion and UI action resolution — no browser needed."""
from __future__ import annotations

import pytest

from gym.actions import (
    Click,
    Navigate,
    Scroll,
    TypeText,
    action_from_oneof,
    action_to_oneof,
    coerce_action,
)
from gym.ui_actions import (
    CategoryFilter,
    FormField,
    SortOption,
    StorePage,
    UiButton,
    available_actions,
    cancel_order,
    click_ui,
    detect_store_page,
    fill_field,
    filter_products,
    goto,
    open_product,
    resolve_ui_action,
    view_order,
)


# ── coerce_action: primitives pass through ────────────────────────────────────

@pytest.mark.parametrize("action", [
    Click("#btn"),
    TypeText("#field", "hello"),
    Scroll(300.0),
    Navigate("/cart"),
])
def test_coerce_passthrough_primitives(action):
    assert coerce_action(action) is action


# ── coerce_action: plain dicts ────────────────────────────────────────────────

def test_coerce_click_dict():
    assert coerce_action({"type": "click", "selector": "#x"}) == Click("#x")


def test_coerce_navigate_dict():
    assert coerce_action({"type": "navigate", "url": "/orders"}) == Navigate("/orders")


def test_coerce_type_dict():
    assert coerce_action({"type": "type", "selector": "#f", "text": "hi"}) == TypeText("#f", "hi")


def test_coerce_scroll_dict():
    assert coerce_action({"type": "scroll", "delta_y": 500}) == Scroll(500.0)


# ── coerce_action: ui-action dicts round-trip through resolve ─────────────────

def test_coerce_goto_dict():
    result = coerce_action({"type": "goto", "page": "orders"})
    assert isinstance(result, Navigate)
    assert result.url == "/orders"


def test_coerce_open_product_dict():
    result = coerce_action({"type": "open_product", "sku": "SKU-E1001"})
    assert isinstance(result, Navigate)
    assert result.url == "/product/SKU-E1001"


def test_coerce_filter_products_dict():
    result = coerce_action({"type": "filter_products", "category": "Electronics", "sort": "price_asc"})
    assert isinstance(result, Navigate)
    assert "category=Electronics" in result.url
    assert "sort=price_asc" in result.url


def test_coerce_ui_click_dict():
    result = coerce_action({"type": "ui_click", "button": "add_to_cart"})
    assert isinstance(result, Click)
    assert "add-to-cart" in result.selector


def test_coerce_ui_fill_dict():
    result = coerce_action({"type": "ui_fill", "field": "shipping_address", "value": "123 Main"})
    assert isinstance(result, TypeText)
    assert "shipping-address" in result.selector
    assert result.text == "123 Main"


def test_coerce_cancel_order_dict():
    result = coerce_action({"type": "cancel_order", "order_id": 3})
    assert isinstance(result, Click)
    assert "cancel-order-3" in result.selector


def test_coerce_view_order_dict():
    result = coerce_action({"type": "view_order", "order_id": 7})
    assert isinstance(result, Click)
    assert "view-order-7" in result.selector


# ── coerce_action: UiAction objects ──────────────────────────────────────────

@pytest.mark.parametrize("ui_action", [
    goto(StorePage.HOME),
    goto(StorePage.ORDERS),
    filter_products(category="Electronics"),
    filter_products(),
    open_product("SKU-E1001"),
    click_ui(UiButton.ADD_TO_CART),
    click_ui(UiButton.PLACE_ORDER),
    fill_field(FormField.SHIPPING_ADDRESS, "addr"),
    fill_field(FormField.QUANTITY, 2),
    view_order(1),
    cancel_order(1),
])
def test_coerce_ui_action_objects(ui_action):
    result = coerce_action(ui_action)
    assert isinstance(result, (Click, Navigate, TypeText, Scroll))


# ── OneOf tuple roundtrip ─────────────────────────────────────────────────────

@pytest.mark.parametrize("action", [
    Click("#x"),
    TypeText("#f", "hello"),
    Scroll(200.0),
    Navigate("/orders"),
])
def test_action_oneof_roundtrip(action):
    assert action_from_oneof(action_to_oneof(action)) == action


# ── resolve_ui_action ─────────────────────────────────────────────────────────

def test_resolve_goto_home():
    assert resolve_ui_action(goto(StorePage.HOME)) == {"type": "navigate", "url": "/"}


def test_resolve_goto_cart():
    assert resolve_ui_action(goto(StorePage.CART)) == {"type": "navigate", "url": "/cart"}


def test_resolve_filter_all_alpha():
    result = resolve_ui_action(filter_products())
    assert result["type"] == "navigate"
    assert "sort=alpha" in result["url"]


def test_resolve_filter_category_and_sort():
    result = resolve_ui_action(filter_products(category="Fitness", sort="price_asc"))
    assert "category=Fitness" in result["url"]
    assert "sort=price_asc" in result["url"]


def test_resolve_open_product():
    assert resolve_ui_action(open_product("SKU-E1001")) == {
        "type": "navigate",
        "url": "/product/SKU-E1001",
    }


def test_resolve_ui_click_all_buttons():
    for btn in UiButton:
        result = resolve_ui_action(click_ui(btn))
        assert result["type"] == "click"
        assert result["selector"]


def test_resolve_ui_fill_quantity():
    result = resolve_ui_action(fill_field(FormField.QUANTITY, 3))
    assert result == {"type": "type", "selector": "[data-testid='quantity']", "text": "3"}


def test_resolve_ui_fill_cart_quantity_requires_sku():
    with pytest.raises(ValueError, match="sku"):
        resolve_ui_action(fill_field(FormField.CART_QUANTITY, 2))


def test_resolve_ui_fill_cart_quantity_with_sku():
    result = resolve_ui_action(fill_field(FormField.CART_QUANTITY, 2, sku="SKU-E1001"))
    assert "cart-quantity-SKU-E1001" in result["selector"]


def test_resolve_cancel_order():
    assert resolve_ui_action(cancel_order(5)) == {
        "type": "click",
        "selector": "[data-testid='cancel-order-5']",
    }


def test_resolve_view_order():
    assert resolve_ui_action(view_order(3)) == {
        "type": "click",
        "selector": "[data-testid='view-order-3']",
    }


# ── available_actions: valid enum values only ─────────────────────────────────

def _assert_valid_action_strings(actions: list[str]) -> None:
    """Every ui_click:X must be a valid UiButton; every ui_fill:X a valid FormField."""
    for a in actions:
        if a.startswith("ui_click:"):
            UiButton(a.split(":", 1)[1])   # raises ValueError if not a valid UiButton
        elif a.startswith("ui_fill:"):
            FormField(a.split(":", 1)[1])  # raises ValueError if not a valid FormField


def test_available_actions_home():
    acts = available_actions("http://localhost:8000/")
    assert "filter_products" in acts["actions"]
    assert "open_product" in acts["actions"]
    _assert_valid_action_strings(acts["actions"])


def test_available_actions_cart():
    acts = available_actions("http://localhost:8000/cart")
    assert "ui_click:checkout_link" in acts["actions"]
    assert "ui_fill:coupon_code" in acts["actions"]
    _assert_valid_action_strings(acts["actions"])


def test_available_actions_product():
    acts = available_actions("http://localhost:8000/product/SKU-E1001")
    assert "ui_click:add_to_cart" in acts["actions"]
    _assert_valid_action_strings(acts["actions"])


def test_available_actions_checkout():
    acts = available_actions("http://localhost:8000/checkout")
    assert "ui_fill:shipping_address" in acts["actions"]
    assert "ui_click:place_order" in acts["actions"]
    _assert_valid_action_strings(acts["actions"])


def test_available_actions_orders():
    acts = available_actions("http://localhost:8000/orders")
    assert "view_order" in acts["actions"]
    assert "cancel_order" in acts["actions"]
    _assert_valid_action_strings(acts["actions"])


def test_available_actions_all_pages_have_goto():
    for page in StorePage:
        url = f"http://localhost:8000/{page.value}".rstrip("/home") or "http://localhost:8000/"
    # All pages should at minimum allow goto
    acts = available_actions("http://localhost:8000/")
    assert "goto" in acts["actions"]


# ── detect_store_page ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("http://localhost:8000/", StorePage.HOME),
    ("http://localhost:8000/cart", StorePage.CART),
    ("http://localhost:8000/orders", StorePage.ORDERS),
    ("http://localhost:8000/checkout", StorePage.CHECKOUT),
    ("http://localhost:8000/product/SKU-E1001", None),
    ("http://localhost:8000/orders/5", None),
    ("http://localhost:8000/orders/5/confirmation", None),
])
def test_detect_store_page(url, expected):
    assert detect_store_page(url) == expected
