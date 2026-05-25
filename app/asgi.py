from __future__ import annotations

import os

from app.main import create_app


seed = int(os.environ.get("WEBGYM_SEED", "0"))
database = os.environ.get("WEBGYM_DB", "data/store.sqlite")

# create_app initializes the DB when missing; gym env resets before each episode.
app = create_app(database, seed)
