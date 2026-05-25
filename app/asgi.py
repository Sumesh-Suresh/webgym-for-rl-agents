from __future__ import annotations

import os

from app import db
from app.main import create_app


seed = int(os.environ.get("WEBGYM_SEED", "0"))
database = os.environ.get("WEBGYM_DB", "data/store.sqlite")
db.init_database(database, seed)

app = create_app(database, seed)
