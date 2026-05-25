from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import db


@pytest.fixture
def tmp_db(tmp_path):
    """Return a callable that seeds a fresh DB at tmp_path and returns its Path."""
    def _make(seed: int):
        path = tmp_path / f"store_{seed}.sqlite"
        db.reset_database(path, seed)
        return path
    return _make
