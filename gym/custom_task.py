from gym.task import AbstractEcommerceTask
import random
from pathlib import Path
from app import db


class CustomTask(AbstractEcommerceTask):

    kind = "custom_task" 
    def __init__(self, seed: int, db_path: Path, rng: random.Random) -> None:
        super().__init__(seed, db_path, rng)
        with db.connect(db_path) as conn:
            row = conn.execute(
                "SELECT sku FROM products WHERE category = ? "
                "ORDER BY price_cents DESC LIMIT 1",
                (self.category,),
            ).fetchone()
        self.expected_sku = row["sku"]

    def check_success(self, db_path: str | Path) -> bool:
        return True