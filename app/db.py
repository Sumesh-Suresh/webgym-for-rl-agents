from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


MIN_PRODUCTS_PER_RESET = 10
MIN_PER_CATEGORY = 2
MIN_ORDERS_PER_RESET = 0
MAX_ORDERS_PER_RESET = 8
MAX_ITEMS_PER_ORDER = 4
CANCEL_RATE = 0.3

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
    """
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0 if seed is None else seed)

    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        product_rows = list(seeded_products(rng))
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

        # NOTE: in the order table created_at is with a base time stamp and iterated with order_id 


def list_products(
    db_path: str | Path,
    category: str | None = None,
    sort_by: str = "alpha",
) -> list[sqlite3.Row]:
    order_by = {
        "alpha": "name COLLATE NOCASE, sku",
        "price_asc": "price_cents, name COLLATE NOCASE, sku",
        "price_desc": "price_cents DESC, name COLLATE NOCASE, sku",
    }.get(sort_by, "name COLLATE NOCASE, sku")
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

        # Coupon lookup shares the same connection and transaction as the order insert.
        coupon_row = (
            conn.execute("SELECT percent_off FROM coupons WHERE code = ?", (coupon_code,)).fetchone()
            if coupon_code
            else None
        )
        if coupon_code and coupon_row is None:
            raise ValueError("invalid coupon")

        percent_off = int(coupon_row["percent_off"]) if coupon_row else 0
        subtotal = sum(products[sku]["price_cents"] * qty for sku, qty in items.items())
        discount = subtotal * percent_off // 100
        total = subtotal - discount

        # Let SQLite assign the order ID via INTEGER PRIMARY KEY autoincrement; use lastrowid.
        cursor = conn.execute(
            """
            INSERT INTO orders (status, shipping_address, coupon_code, total_cents, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("placed", shipping_address, coupon_code, total, current_created_at()), # added by policy or manually from UI
        )
        order_id = cursor.lastrowid
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
    by_category: dict[str, list[tuple[str, str, str, int, str]]] = {}
    for product in BASE_PRODUCTS:
        by_category.setdefault(product[2], []).append(product)

    selected: list[tuple[str, str, str, int, str]] = []
    leftovers: list[tuple[str, str, str, int, str]] = []

    for category in sorted(by_category):
        bucket = list(by_category[category])
        rng.shuffle(bucket)
        selected.extend(bucket[:MIN_PER_CATEGORY])   # guarantee ≥ MIN_PER_CATEGORY per category
        leftovers.extend(bucket[MIN_PER_CATEGORY:])

    rng.shuffle(leftovers)

    deficit = max(0, MIN_PRODUCTS_PER_RESET - len(selected))
    selected.extend(leftovers[:deficit])

    selected.sort(key=lambda product: product[0])  # stable SKU order

    for product_id, (sku, name, category, price_cents, description) in enumerate(selected, start=1):
        yield (product_id, sku, name, category, price_cents, description)


def seeded_orders(
    rng: random.Random,
    product_rows: Iterable[tuple[int, str, str, str, int, str]],
) -> tuple[list[tuple[int, str, str, str | None, int, str]], list[tuple[int, str, int, int]]]:
    products = list(product_rows)
    n_orders = rng.randint(MIN_ORDERS_PER_RESET, MAX_ORDERS_PER_RESET)

    order_rows: list[tuple[int, str, str, str | None, int, str]] = []
    order_item_rows: list[tuple[int, str, int, int]] = []

    base_date = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc) # for deterministic timestamps

    for order_id in range(1, n_orders + 1):
        item_count = rng.randint(1, min(2, len(products)))
        order_products = rng.sample(products, item_count)
        coupon_code = rng.choice([None] + sorted(COUPON_CODE_MAP.keys()))
        status = "cancelled" if rng.random() < CANCEL_RATE else "placed"
        created_at = (base_date + timedelta(days=order_id - 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        shipping_address = rng.choice(ADDRESS_LIST)

        subtotal = 0
        for _, sku, _, _, price_cents, _ in order_products:
            quantity = rng.randint(1, MAX_ITEMS_PER_ORDER)
            order_item_rows.append((order_id, sku, quantity, price_cents))
            subtotal += price_cents * quantity

        discount = subtotal * COUPON_CODE_MAP[coupon_code] // 100 if coupon_code else 0
        total_cents = subtotal - discount
        order_rows.append((order_id, status, shipping_address, coupon_code, total_cents, created_at))

    return order_rows, order_item_rows


def current_created_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
