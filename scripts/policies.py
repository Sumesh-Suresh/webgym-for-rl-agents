"""A trivial random-clicking policy used as a low baseline in the parallel
rollout demo. It picks a random clickable / fillable element from the page,
or occasionally navigates to a top-level URL. It is intentionally bad — we
expect it to almost never succeed at any of the three tasks. Its purpose is
to confirm that verifiers don't false-positive on accidental success."""

from __future__ import annotations

import random
from typing import Optional


_TOP_LEVEL_URLS = ["/", "/cart", "/checkout", "/orders"]


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def act(self, obs: dict) -> dict:
        roll = self._rng.random()
        if roll < 0.15:
            return {"type": "navigate", "url": self._rng.choice(_TOP_LEVEL_URLS)}
        if roll < 0.25:
            return {"type": "scroll", "dy": self._rng.choice([-400, 400])}

        selector = self._pick_selector(obs.get("dom", ""))
        if selector is None:
            return {"type": "noop"}

        # If it looks like a text input, type something silly.
        if "data-testid=\"shipping-address\"" in selector or "name=\"coupon_code\"" in selector:
            return {"type": "type", "selector": selector, "text": self._rng.choice(["xxx", "asdf", ""])}
        if "name=\"quantity\"" in selector:
            return {"type": "type", "selector": selector, "text": str(self._rng.randint(1, 5))}

        return {"type": "click", "selector": selector}

    # ------- helpers ---------------------------------------------------

    def _pick_selector(self, dom: str) -> Optional[str]:
        # Lightweight DOM scrape: collect data-testid values present on the page
        # and choose one at random. Avoids importing a parser into the policy.
        import re

        ids = re.findall(r'data-testid="([^"]+)"', dom)
        if not ids:
            return None
        return f'[data-testid="{self._rng.choice(ids)}"]'
