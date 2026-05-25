from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


MIN_PRODUCTS_PER_RESET = 10
MIN_ORDERS_PER_RESET = 0
MAX_ORDERS_PER_RESET = 8

BASE_PRODUCTS = [
    ("SKU-E1001", "USB-C Charging Cable", "Electronics", 1299, "Braided 6ft cable."),
    ("SKU-E2042", "Bluetooth Speaker", "Electronics", 3499, "Portable speaker."),
    ("SKU-E7421", "Noise-Canceling Headphones", "Electronics", 8999, "Over-ear headphones."),
    ("SKU-E3333", "Wireless Mouse", "Electronics", 2499, "Ergonomic mouse."),
    ("SKU-H1001", "Ceramic Mug", "Home", 1099, "Dishwasher-safe mug."),
    ("SKU-H2002", "Desk Lamp", "Home", 2999, "LED task lamp."),
    ("SKU-H3003", "Throw Blanket", "Home", 3999, "Soft woven blanket."),
    ("SKU-H4004", "Storage Basket", "Home", 2299, "Woven organizer basket."),
    ("SKU-H5005", "Wall Clock", "Home", 1899, "Minimal analog clock."),
    ("SKU-F1001", "Running Socks", "Fitness", 999, "Cushioned socks."),
    ("SKU-F2002", "Yoga Mat", "Fitness", 2799, "Non-slip mat."),
    ("SKU-F3003", "Resistance Bands", "Fitness", 1999, "Set of three bands."),
    ("SKU-F4004", "Foam Roller", "Fitness", 2499, "Textured recovery roller."),
    ("SKU-F5005", "Water Bottle", "Fitness", 1599, "Insulated stainless bottle."),
    ("SKU-O1001", "Notebook", "Office", 699, "Ruled pages."),
    ("SKU-O2002", "Ballpoint Pens", "Office", 499, "Pack of ten."),
    ("SKU-O3003", "Desk Organizer", "Office", 1799, "Multi-compartment organizer."),
    ("SKU-O4004", "Sticky Notes", "Office", 599, "Assorted color notes."),
    ("SKU-O5005", "Monitor Stand", "Office", 3299, "Raised desktop stand."),
]
COUPON_CODE_MAP = {"SAVE10": 10}

ADDRESS_LIST = [
    "123 Main St, Springfield, IL 62701",
    "456 Oak Ave, Metropolis, NY 10001",
    "789 Pine St, Gotham City, NY 10001",
]

SCHEMA = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS coupons;
DROP TABLE IF EXISTS products;

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price_cents INTEGER NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE coupons (
    code TEXT PRIMARY KEY,
    percent_off INTEGER NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    shipping_address TEXT NOT NULL,
    coupon_code TEXT,
    total_cents INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (coupon_code) REFERENCES coupons(code)
);

CREATE TABLE order_items (
    order_id INTEGER NOT NULL,
    sku TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price_cents INTEGER NOT NULL,
    PRIMARY KEY (order_id, sku),
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (sku) REFERENCES products(sku)
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def reset_database(db_path: str | Path, seed: int | None = None) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0 if seed is None else seed)

    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        # Build the product seed rows once so orders can reference the same sampled SKUs.
        product_rows = list(seeded_products(rng))
        # Build compatible seeded orders and order item rows from the sampled products.
        order_rows, order_item_rows = seeded_orders(rng, product_rows)
        conn.executemany(
            """
            INSERT INTO products (id, sku, name, category, price_cents, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            product_rows,
        )
        conn.executemany(
            "INSERT INTO coupons (code, percent_off) VALUES (?, ?)",
            sorted(COUPON_CODE_MAP.items()),
        )
        conn.executemany(
            """
            INSERT INTO orders (id, status, shipping_address, coupon_code, total_cents, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            order_rows,
        )
        conn.executemany(
            """
            INSERT INTO order_items (order_id, sku, quantity, unit_price_cents)
            VALUES (?, ?, ?, ?)
            """,
            order_item_rows,
        )


def list_products(
    db_path: str | Path,
    category: str | None = None,
    sort_by: str = "category",
) -> list[sqlite3.Row]:
    order_by = {
        "alpha": "name COLLATE NOCASE, sku",
        "price_asc": "price_cents, name COLLATE NOCASE, sku",
        "price_desc": "price_cents DESC, name COLLATE NOCASE, sku",
    }.get(sort_by, "category, price_cents, sku")
    with connect(db_path) as conn:
        if category:
            return conn.execute(
                f"SELECT * FROM products WHERE category = ? ORDER BY {order_by}",
                (category,),
            ).fetchall()
        return conn.execute(f"SELECT * FROM products ORDER BY {order_by}").fetchall()


def get_product(db_path: str | Path, sku: str) -> sqlite3.Row | None:
    with connect(db_path) as conn:
        return conn.execute("SELECT * FROM products WHERE sku = ?", (sku,)).fetchone()


def ensure_products(db_path: str | Path, skus: Iterable[str]) -> None:
    """Insert catalog products that are missing from a seeded database.

    The storefront reset samples a subset of BASE_PRODUCTS; gym tasks that
    reference specific SKUs call this before creating orders."""
    catalog = {product[0]: product for product in BASE_PRODUCTS}
    with connect(db_path) as conn:
        for sku in skus:
            if conn.execute("SELECT 1 FROM products WHERE sku = ?", (sku,)).fetchone():
                continue
            product = catalog.get(sku)
            if product is None:
                raise ValueError(f"unknown sku: {sku}")
            _, name, category, price_cents, description = product
            next_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM products").fetchone()[0]
            conn.execute(
                """
                INSERT INTO products (id, sku, name, category, price_cents, description)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (next_id, sku, name, category, price_cents, description),
            )


def get_coupon(db_path: str | Path, code: str | None) -> sqlite3.Row | None:
    if not code:
        return None
    with connect(db_path) as conn:
        return conn.execute("SELECT * FROM coupons WHERE code = ?", (code,)).fetchone()


def create_order(
    db_path: str | Path,
    items: dict[str, int],
    shipping_address: str,
    coupon_code: str | None,
) -> int:
    if not items:
        raise ValueError("cart is empty")

    with connect(db_path) as conn:
        product_rows = conn.execute(
            f"SELECT * FROM products WHERE sku IN ({','.join('?' for _ in items)})",
            tuple(items.keys()),
        ).fetchall()
        products = {row["sku"]: row for row in product_rows}
        missing = set(items) - set(products)
        if missing:
            raise ValueError(f"unknown SKU(s): {', '.join(sorted(missing))}")

        subtotal = sum(products[sku]["price_cents"] * qty for sku, qty in items.items())
        coupon = get_coupon(db_path, coupon_code)
        if coupon_code and coupon is None:
            raise ValueError("invalid coupon")
        discount = subtotal * (coupon["percent_off"] if coupon else 0) // 100
        total = subtotal - discount
        order_id = _next_order_id(conn)
        created_at = current_created_at()
        conn.execute(
            """
            INSERT INTO orders (id, status, shipping_address, coupon_code, total_cents, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (order_id, "placed", shipping_address, coupon_code, total, created_at),
        )
        conn.executemany(
            """
            INSERT INTO order_items (order_id, sku, quantity, unit_price_cents)
            VALUES (?, ?, ?, ?)
            """,
            [
                (order_id, sku, quantity, products[sku]["price_cents"])
                for sku, quantity in sorted(items.items())
            ],
        )
        return order_id


def cancel_order(db_path: str | Path, order_id: int) -> bool:
    with connect(db_path) as conn:
        result = conn.execute(
            "UPDATE orders SET status = 'cancelled' WHERE id = ? AND status = 'placed'",
            (order_id,),
        )
        return result.rowcount == 1


def list_orders(db_path: str | Path) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute("SELECT * FROM orders ORDER BY created_at DESC, id DESC").fetchall()


def get_order(db_path: str | Path, order_id: int) -> sqlite3.Row | None:
    with connect(db_path) as conn:
        return conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()


def get_order_items(db_path: str | Path, order_id: int) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM order_items WHERE order_id = ? ORDER BY sku",
            (order_id,),
        ).fetchall()


def get_order_contents(db_path: str | Path, order_id: int) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            """
            SELECT
                oi.sku,
                oi.quantity,
                oi.unit_price_cents,
                p.name,
                oi.quantity * oi.unit_price_cents AS line_total
            FROM order_items oi
            JOIN products p ON p.sku = oi.sku
            WHERE oi.order_id = ?
            ORDER BY oi.sku
            """,
            (order_id,),
        ).fetchall()



def cheapest_product_in_category(db_path: str | Path, category: str) -> sqlite3.Row:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM products WHERE category = ? ORDER BY price_cents, sku LIMIT 1",
            (category,),
        ).fetchone()
    if row is None:
        raise ValueError(f"category has no products: {category}")
    return row


def seeded_products(rng: random.Random) -> Iterable[tuple[int, str, str, str, int, str]]:
    # Bucket the catalog by category so we can guarantee at least one pick per category.
    by_category: dict[str, list[tuple[str, str, str, int, str]]] = {}
    for product in BASE_PRODUCTS:
        by_category.setdefault(product[2], []).append(product)

    # Holds the products we will actually insert into the table.
    selected: list[tuple[str, str, str, int, str]] = []
    # Pool of remaining products we can draw from to reach MIN_PRODUCTS_PER_RESET.
    leftovers: list[tuple[str, str, str, int, str]] = []

    # Iterate categories in a fixed (sorted) order so shuffles are reproducible per seed.
    for category in sorted(by_category):
        # Copy the bucket before shuffling so the module-level BASE_PRODUCTS stays untouched.
        bucket = list(by_category[category])
        # Deterministic shuffle driven by the seeded rng provided by the caller.
        rng.shuffle(bucket)
        # First item after shuffling becomes the category's guaranteed representative.
        selected.append(bucket[0])
        # Remaining items in this category join the pool for optional backfill.
        leftovers.extend(bucket[1:])

    # Shuffle the leftover pool once so any extra picks are also deterministic per seed.
    rng.shuffle(leftovers)

    # If category coverage alone did not hit the minimum, top up from the shuffled pool.
    deficit = max(0, MIN_PRODUCTS_PER_RESET - len(selected))
    selected.extend(leftovers[:deficit])

    # Sort the final selection by SKU so assigned primary-key ids are stable per seed.
    selected.sort(key=lambda product: product[0])

    # Emit rows in the exact column order expected by the INSERT statement.
    for product_id, (sku, name, category, price_cents, description) in enumerate(selected, start=1):
        yield (product_id, sku, name, category, price_cents, description)


def seeded_orders(
    rng: random.Random,
    product_rows: Iterable[tuple[int, str, str, str, int, str]],
) -> tuple[list[tuple[int, str, str, str | None, int, str]], list[tuple[int, str, int, int]]]:
    # Convert the iterable to a list so we can sample from the exact products inserted above.
    products = list(product_rows)
    # Randomize order candidates deterministically using the same seeded rng as product seeding.
    rng.shuffle(products)

    # Draw the order count from the seeded rng so different seeds produce different history lengths.
    order_count = rng.randint(MIN_ORDERS_PER_RESET, min(MAX_ORDERS_PER_RESET, len(products)))
    # Store rows for the orders table.
    order_rows: list[tuple[int, str, str, str | None, int, str]] = []
    # Store rows for the order_items table.
    order_item_rows: list[tuple[int, str, int, int]] = []

    # Create one seeded order at a time using slices of the shuffled product list.
    for order_id in range(1, order_count + 1):
        # Pick one or two products per order without referencing products outside the seeded table.
        item_count = min(rng.randint(1, 2), len(products))
        # Rotate through the shuffled products so each order starts from a deterministic position.
        start_index = (order_id - 1) * item_count % len(products)
        order_products = [products[(start_index + offset) % len(products)] for offset in range(item_count)]

        # Apply the seeded coupon to every other seeded order so coupon totals are exercised.
        coupon_code = next(iter(COUPON_CODE_MAP)) if order_id % 2 == 0 else None
        # Keep earlier seeded orders cancelable and include one cancelled example for history views.
        # status = "cancelled" if order_id == 1 else "placed" TODO: add cancelled for random seeding
        status = "cancelled" if rng.random() < 0.5 else "placed"
        # Spread order dates one day apart starting from 2024-01-01 so history is always readable.
        _base_date = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        created_at = (_base_date + timedelta(days=order_id - 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        # created_at = current_created_at()

        # Track the subtotal before any coupon discount.
        subtotal = 0
        for _, sku, _, _, price_cents, _ in order_products:
            # Quantity is deterministic and intentionally small for readable seeded orders.
            quantity = rng.randint(1, 2)
            # Record the order_items row using the product's seeded SKU and price.
            order_item_rows.append((order_id, sku, quantity, price_cents))
            # Add this line item to the order subtotal.
            subtotal += price_cents * quantity

        # Apply the same coupon percentage used by the coupons table.
        discount = subtotal * COUPON_CODE_MAP[coupon_code] // 100 if coupon_code else 0
        # Store the final order total after any coupon discount.
        total_cents = subtotal - discount
        # Pick a seeded shipping address from the configured address list.
        shipping_address = rng.choice(ADDRESS_LIST)
        # Record the orders row in the exact column order expected by the INSERT statement.
        order_rows.append((order_id, status, shipping_address, coupon_code, total_cents, created_at))

    # Return both related table payloads together so reset_database can insert them in FK order.
    return order_rows, order_item_rows



def _next_order_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM orders").fetchone()
    return int(row["next_id"])


def current_created_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
