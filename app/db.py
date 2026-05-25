from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_ADDRESS = "123 Main St, Springfield, IL 62701"


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


INIT_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price_cents INTEGER NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS coupons (
    code TEXT PRIMARY KEY,
    percent_off INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    shipping_address TEXT NOT NULL,
    coupon_code TEXT,
    total_cents INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (coupon_code) REFERENCES coupons(code)
);

CREATE TABLE IF NOT EXISTS order_items (
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


def init_database(db_path: str | Path, seed: int | None = None) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0 if seed is None else seed)

    with connect(db_path) as conn:
        conn.executescript(INIT_SCHEMA)
        product_count = conn.execute("SELECT COUNT(*) AS n FROM products").fetchone()["n"]
        if product_count == 0:
            conn.executemany(
                """
                INSERT INTO products (id, sku, name, category, price_cents, description)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                _seeded_products(rng),
            )
            conn.execute(
                "INSERT OR IGNORE INTO coupons (code, percent_off) VALUES (?, ?)",
                ("SAVE10", 10),
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


def _seeded_products(rng: random.Random) -> Iterable[tuple[int, str, str, int, str]]:
    for product_id, (sku, name, category, base_price, description) in enumerate(BASE_PRODUCTS, 1):
        variation = rng.randint(-100, 100)
        price = max(199, base_price + variation)
        if sku == "SKU-E1001":
            price = 999
        if sku == "SKU-E7421":
            price = 8999
        yield product_id, sku, name, category, price, description  # can append and then return list (in-memory)

# IF YOU NEED SELECTION IN PRODUCTS
# def _seeded_products(
#     rng: random.Random,
# ) -> Iterable[tuple[int, str, str, int, str]]:

#     products = list(BASE_PRODUCTS)

#     # deterministic shuffle
#     rng.shuffle(products)

#     # keep guaranteed SKUs needed for tasks
#     required_skus = {"SKU-E1001", "SKU-E7421"}

#     selected = []

#     # always include required products
#     for p in products:
#         if p[0] in required_skus:
#             selected.append(p)

#     # deterministic category balancing
#     categories = {
#         "Electronics": 4,
#         "Home": 3,
#         "Fitness": 2,
#         "Office": 2,
#     }

#     for category, limit in categories.items():
#         category_products = [
#             p for p in products
#             if p[2] == category and p not in selected
#         ]

#         selected.extend(category_products[:limit])

#     # deterministic ordering
#     selected.sort(key=lambda p: p[0])

#     for product_id, (sku, name, category, base_price, description) in enumerate(selected, 1):

#         # deterministic price variation
#         variation = rng.randint(-300, 300)
#         price = max(199, base_price + variation)

#         # enforce task invariants
#         if sku == "SKU-E1001":
#             price = 999

#         if sku == "SKU-E7421":
#             price = 8999

#         yield (
#             product_id,
#             sku,
#             name,
#             category,
#             price,
#             description,
#         )

def _next_order_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 AS next_id FROM orders").fetchone()
    return int(row["next_id"])


def current_created_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
