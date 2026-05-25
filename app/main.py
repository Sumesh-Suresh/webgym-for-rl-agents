from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app import db


CATEGORIES = ("Electronics", "Fitness", "Home", "Office")
SORT_OPTIONS = {
    "alpha": "Alphabetical",
    "price_asc": "Price: low to high",
    "price_desc": "Price: high to low",
}


def create_app(db_path: str | Path | None = None, seed: int | None = None) -> FastAPI:
    database = Path(db_path or os.environ.get("WEBGYM_DB", "data/store.sqlite"))
    if not database.exists():
        db.reset_database(database, seed)

    app = FastAPI(title="Minimal E-commerce Gym Storefront")
    app.state.db_path = str(database)

    @app.get("/", response_class=HTMLResponse)
    async def index(category: Optional[str] = None, sort: str = "alpha"):
        selected_category = category if category in CATEGORIES else None
        selected_sort = sort if sort in SORT_OPTIONS else "alpha"
        products = db.list_products(app.state.db_path, category=selected_category, sort_by=selected_sort)
        category_options = "\n".join(
            f'<option value="{escape(name)}"{selected(selected_category, name)}>{escape(name)}</option>'
            for name in CATEGORIES
        )
        sort_options = "\n".join(
            f'<option value="{escape(value)}"{selected(selected_sort, value)}>{escape(label)}</option>'
            for value, label in SORT_OPTIONS.items()
        )
        items = "\n".join(
            f"""
            <tr data-testid="product-{escape(product['sku'])}">
              <td><a data-oracle-action="product-link" href="/product/{escape(product['sku'])}">{escape(product['name'])}</a></td>
              <td>{escape(product['sku'])}</td>
              <td>{money(product['price_cents'])}</td>
            </tr>
            """
            for product in products
        )
        return page(
            "Stationary Shop",
            f"""
            <h1>Stationary Shop</h1>
            <nav class="home-nav" aria-label="Store pages">
              <a href="/">Products</a>
              <a href="/cart">Cart</a>
              <a href="/checkout">Checkout</a>
              <a href="/orders">Order History</a>
            </nav>
            <form class="product-filters" method="get" action="/" aria-label="Product filters">
              <label>Category
                <select data-testid="category-filter" name="category">
                  <option value="">All</option>
                  {category_options}
                </select>
              </label>
              <label>Sort by
                <select data-testid="sort-filter" name="sort">
                  {sort_options}
                </select>
              </label>
              <button data-testid="apply-filters" type="submit">Apply filters</button>
            </form>
            <table aria-label="Product list">
              <thead>
                <tr>
                  <th scope="col">Product</th>
                  <th scope="col">Product ID</th>
                  <th scope="col">Price</th>
                </tr>
              </thead>
              <tbody>{items}</tbody>
            </table>
            """,
            show_home_link=False,
        )

    @app.get("/product/{sku}", response_class=HTMLResponse)
    async def product_detail(sku: str):
        product = db.get_product(app.state.db_path, sku)
        if product is None:
            raise HTTPException(status_code=404)
        return page(
            product["name"],
            f"""
            <h1>{escape(product['name'])}</h1>
            <p data-testid="sku">{escape(product['sku'])}</p>
            <p>{escape(product['description'])}</p>
            <p data-testid="price">{money(product['price_cents'])}</p>
            <form method="post" action="/cart/add">
              <input type="hidden" name="sku" value="{escape(product['sku'])}">
              <label>Quantity <input data-testid="quantity" name="quantity" type="number" min="1" value="1"></label>
              <button data-testid="add-to-cart" type="submit">Add to cart</button>
            </form>
            """,
        )

    @app.post("/cart/add")
    async def add_to_cart(request: Request):
        form = await read_form(request)
        sku = form.get("sku", "")
        quantity = parse_quantity(form.get("quantity", "1"))
        if db.get_product(app.state.db_path, sku) is None:
            raise HTTPException(status_code=404)
        cart = read_cart(request)
        cart[sku] = cart.get(sku, 0) + quantity
        response = redirect("/cart")
        write_cart(response, cart)
        return response

    @app.get("/cart", response_class=HTMLResponse)
    async def cart(request: Request):
        rows, subtotal = cart_rows(app.state.db_path, read_cart(request))
        coupon_code = request.cookies.get("coupon")
        coupon = db.get_coupon(app.state.db_path, coupon_code)
        discount = subtotal * (coupon["percent_off"] if coupon else 0) // 100
        if rows:
            item_html = "\n".join(
                f"""
                <tr data-testid="cart-item-{escape(item['sku'])}">
                  <td>{escape(item['name'])}</td>
                  <td>{escape(item['sku'])}</td>
                  <td>
                    <form class="cart-action" method="post" action="/cart/update">
                      <input type="hidden" name="sku" value="{escape(item['sku'])}">
                      <input
                        data-testid="cart-quantity-{escape(item['sku'])}"
                        name="quantity"
                        type="number"
                        min="1"
                        value="{item['quantity']}"
                      >
                      <button data-testid="update-cart-item-{escape(item['sku'])}" type="submit">Update</button>
                    </form>
                  </td>
                  <td>{money(item['line_total'])}</td>
                  <td>
                    <form class="cart-action" method="post" action="/cart/remove">
                      <input type="hidden" name="sku" value="{escape(item['sku'])}">
                      <button data-testid="remove-cart-item-{escape(item['sku'])}" type="submit">Remove</button>
                    </form>
                  </td>
                </tr>
                """
                for item in rows
            )
            coupon_html = (
                f'<p data-testid="applied-coupon">Coupon {escape(coupon_code)}</p>' if coupon_code else ""
            )
            body = f"""
            <h1>Cart</h1>
            <table aria-label="Cart items">
              <thead>
                <tr>
                  <th scope="col">Product</th>
                  <th scope="col">Product ID</th>
                  <th scope="col">Quantity</th>
                  <th scope="col">Total</th>
                  <th scope="col">Action</th>
                </tr>
              </thead>
              <tbody>{item_html}</tbody>
              <tfoot>
                <tr>
                  <td colspan="5">
                    <a href="/">Add More</a>
                    <a data-testid="checkout-link" href="/checkout">Checkout</a>
                  </td>
                </tr>
              </tfoot>
            </table>
            <p>Subtotal {money(subtotal)}</p>
            {coupon_html}
            <p>Total {money(subtotal - discount)}</p>
            <form method="post" action="/cart/coupon">
              <label>Coupon <input data-testid="coupon-code" name="coupon_code" value="{escape(coupon_code or '')}"></label>
              <button data-testid="apply-coupon" type="submit">Apply coupon</button>
            </form>
            """
        else:
            body = "<h1>Cart</h1><p>Your cart is empty.</p>"
        return page("Cart", body, show_home_link=False)

    @app.post("/cart/update")
    async def update_cart(request: Request):
        form = await read_form(request)
        sku = form.get("sku", "")
        if db.get_product(app.state.db_path, sku) is None:
            raise HTTPException(status_code=404)
        cart = read_cart(request)
        if sku in cart:
            cart[sku] = parse_quantity(form.get("quantity", "1"))
        response = redirect("/cart")
        write_cart(response, cart)
        return response

    @app.post("/cart/remove")
    async def remove_from_cart(request: Request):
        form = await read_form(request)
        cart = read_cart(request)
        cart.pop(form.get("sku", ""), None)
        response = redirect("/cart")
        write_cart(response, cart)
        return response

    @app.post("/cart/coupon")
    async def apply_coupon(request: Request):
        form = await read_form(request)
        code = form.get("coupon_code", "").strip().upper()
        response = redirect("/cart")
        if db.get_coupon(app.state.db_path, code) is not None:
            response.set_cookie("coupon", code, httponly=True, samesite="lax")
        return response

    @app.get("/checkout", response_class=HTMLResponse)
    async def checkout(request: Request):
        if not read_cart(request):
            return redirect("/cart")
        return page(
            "Checkout",
            """
            <h1>Checkout</h1>
            <form method="post" action="/checkout">
              <label>Shipping address
                <textarea data-testid="shipping-address" name="shipping_address"></textarea>
              </label>
              <button data-testid="place-order" type="submit">Place order</button>
            </form>
            """,
        )

    @app.post("/checkout")
    async def place_order(request: Request):
        cart = read_cart(request)
        if not cart:
            return redirect("/cart")
        form = await read_form(request)
        address = form.get("shipping_address", "").strip()
        if not address:
            return redirect("/checkout")
        order_id = db.create_order(app.state.db_path, cart, address, request.cookies.get("coupon"))
        response = redirect(f"/orders/{order_id}/confirmation")
        write_cart(response, {})
        response.delete_cookie("coupon")
        return response

    @app.get("/orders/{order_id}/confirmation", response_class=HTMLResponse)
    async def confirmation(order_id: int):
        return page(
            "Order confirmation",
            f"""
            <h1>Order confirmation</h1>
            <p data-testid="order-id">Order {order_id}</p>
            <a href="/orders">View orders</a>
            """,
        )

    @app.get("/orders", response_class=HTMLResponse)
    async def orders():
        order_items = "\n".join(
            f"""
            <li data-testid="order-{order['id']}">
              Order {order['id']} {escape(order['status'])} {escape(format_order_date(order['created_at']))}
              {cancel_form(order['id']) if order['status'] == 'placed' else ''}
              <a data-testid="view-order-{order['id']}" data-oracle-action="view-order" href="/orders/{order['id']}">View Order</a>
            </li>
            """
            for order in db.list_orders(app.state.db_path)
        )
        return page("Orders", f"<h1>Orders</h1><ul aria-label=\"Orders\">{order_items}</ul>")

    @app.get("/orders/{order_id}", response_class=HTMLResponse)
    async def order_detail(order_id: int):
        order = db.get_order(app.state.db_path, order_id)
        if order is None:
            raise HTTPException(status_code=404)
        rows = db.get_order_contents(app.state.db_path, order_id)
        item_rows = "\n".join(
            f"""
            <tr data-testid="order-item-{escape(item['sku'])}">
              <td>{escape(item['name'])}</td>
              <td>{escape(item['sku'])}</td>
              <td>{item['quantity']}</td>
              <td>{money(item['unit_price_cents'])}</td>
              <td>{money(item['line_total'])}</td>
            </tr>
            """
            for item in rows
        )
        subtotal = sum(item["line_total"] for item in rows)
        coupon = db.get_coupon(app.state.db_path, order["coupon_code"])
        discount = subtotal - order["total_cents"]
        discount_html = ""
        if coupon is not None and discount > 0:
            discount_html = f"""
            <p data-testid="order-coupon">Coupon {escape(order['coupon_code'])} ({coupon['percent_off']}% off)</p>
            <p data-testid="order-discount">Discount -{money(discount)}</p>
            """
        return page(
            f"Order {order_id}",
            f"""
            <h1>Order {order_id}</h1>
            <p>{escape(order['status'])}</p>
            <p>{escape(format_order_date(order['created_at']))}</p>
            <table aria-label="Order contents">
              <thead>
                <tr>
                  <th scope="col">Product</th>
                  <th scope="col">Product ID</th>
                  <th scope="col">Quantity</th>
                  <th scope="col">Unit Price</th>
                  <th scope="col">Total</th>
                </tr>
              </thead>
              <tbody>{item_rows}</tbody>
            </table>
            <p>Subtotal {money(subtotal)}</p>
            {discount_html}
            <p>Order total {money(order['total_cents'])}</p>
            <a href="/orders">Back to order history</a>
            """,
        )

    @app.post("/orders/{order_id}/cancel")
    async def cancel_order(order_id: int):
        db.cancel_order(app.state.db_path, order_id)
        return redirect("/orders")

    return app


async def read_form(request: Request) -> dict[str, str]:
    body = (await request.body()).decode()
    parsed = parse_qs(body, keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items()}


def read_cart(request: Request) -> dict[str, int]:
    raw = request.cookies.get("cart")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return {str(sku): int(quantity) for sku, quantity in parsed.items()}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def write_cart(response: RedirectResponse, cart: dict[str, int]) -> None:
    if cart:
        response.set_cookie("cart", json.dumps(cart, separators=(",", ":")), httponly=True, samesite="lax")
    else:
        response.delete_cookie("cart")


def cart_rows(db_path: str | Path, cart: dict[str, int]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    subtotal = 0
    for sku, quantity in sorted(cart.items()):
        product = db.get_product(db_path, sku)
        if product is None:
            continue
        line_total = product["price_cents"] * quantity
        subtotal += line_total
        rows.append(
            {
                "sku": sku,
                "name": product["name"],
                "quantity": quantity,
                "line_total": line_total,
            }
        )
    return rows, subtotal


def page(title: str, body: str, show_home_link: bool = True) -> HTMLResponse:
    home_link = '<nav class="site-nav"><a href="/">Home</a></nav>' if show_home_link else ""
    return HTMLResponse(
        f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>{escape(title)}</title>
          <style>
            .home-nav {{
              display: flex;
              gap: 1rem;
              margin-top: 1rem;
            }}

            .site-nav {{
              margin-bottom: 1.5rem;
            }}

            .product-filters {{
              margin-top: 2rem;
              margin-bottom: 1.5rem;
            }}

            table[aria-label="Product list"] {{
              border-collapse: collapse;
              margin-top: 1rem;
            }}

            table[aria-label="Cart items"] {{
              border-collapse: collapse;
              margin: 1rem 0;
            }}

            table[aria-label="Order contents"] {{
              border-collapse: collapse;
              margin: 1rem 0;
            }}

            .order-action {{
              display: inline;
              margin: 0 0.5rem;
            }}

            table[aria-label="Cart items"] tfoot a {{
              margin-right: 1rem;
            }}

            .cart-action {{
              display: inline-flex;
              gap: 0.5rem;
              align-items: center;
            }}

            .cart-action input[name="quantity"] {{
              width: 4rem;
            }}

            table[aria-label="Product list"] th,
            table[aria-label="Product list"] td,
            table[aria-label="Cart items"] th,
            table[aria-label="Cart items"] td,
            table[aria-label="Order contents"] th,
            table[aria-label="Order contents"] td {{
              border: 1px solid #999;
              padding: 0.5rem 1rem;
              text-align: left;
            }}
          </style>
        </head>
        <body>
          {home_link}
          <main>{body}</main>
        </body>
        </html>
        """
    )


def redirect(path: str) -> RedirectResponse:
    return RedirectResponse(path, status_code=303)


def cancel_form(order_id: int) -> str:
    return f"""
    <form class="order-action" method="post" action="/orders/{order_id}/cancel">
      <button data-testid="cancel-order-{order_id}" data-oracle-action="cancel-order" type="submit">Cancel order</button>
    </form>
    """


def selected(current: object, value: object) -> str:
    return " selected" if current == value else ""


def parse_quantity(raw: str) -> int:
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def format_order_date(created_at: str) -> str:
    parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    eastern = parsed.astimezone(ZoneInfo("America/New_York"))
    return f"Placed on {ordinal(eastern.day)} {eastern:%B %Y %-I:%M %p} ET"


def ordinal(day: int) -> str:
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def money(cents: int) -> str:
    return f"${cents / 100:.2f}"


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)
