"""Gymnasium-compatible web env for the e-commerce gym.

Each EcomEnv instance owns:
  * a private SQLite file (in a tempdir),
  * a private uvicorn process serving the FastAPI app on a free port,
  * a private Playwright Chromium browser context.

This guarantees parallel safety by isolation — no shared mutable state between
instances. Reset re-seeds the SQLite file and reloads the start URL; the server
process stays up across resets (only resetting the DB and reopening the
browser context) so steady-state reset latency stays well under the 3s budget.

Observation:
    {
        "url":        str,
        "dom":        str   (full HTML),
        "axtree":     list  (browser accessibility snapshot),
        "screenshot": bytes (PNG),
    }

Action (dict):
    {"type": "click",     "selector": "<css>"}
    {"type": "type",      "selector": "<css>", "text": "<value>", "clear": true}
    {"type": "scroll",    "dx": 0, "dy": 500}
    {"type": "navigate",  "url": "/orders"}            # relative or absolute
    {"type": "noop"}                                    # do nothing
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Any, Optional

from app import db
from gym.tasks import SNAPSHOTS, Task, get_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_http(url: str, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if 200 <= resp.status < 500:
                    return
        except Exception as e:                                # noqa: BLE001
            last_err = e
            time.sleep(0.05)
    raise RuntimeError(f"server at {url} did not start in {timeout_s}s ({last_err!r})")


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class EcomEnv:
    """Gymnasium-style environment. Not a subclass of `gymnasium.Env` to avoid a
    hard dep, but the surface (reset / step / close, returning the standard
    5-tuple) is compatible."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task_id: str,
        max_steps: int = 50,
        headless: bool = True,
        screenshot: bool = True,
        keep_workdir: bool = False,
    ) -> None:
        self.task: Task = get_task(task_id)
        self.max_steps = max_steps
        self.headless = headless
        self.screenshot_enabled = screenshot
        self.keep_workdir = keep_workdir

        self._workdir = Path(tempfile.mkdtemp(prefix="ecomgym-"))
        self._db_path = self._workdir / "store.sqlite"
        self._port = _free_port()
        self._base_url = f"http://127.0.0.1:{self._port}"
        self._server: Optional[subprocess.Popen] = None

        # Playwright handles, lazily created on first reset.
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

        self._step_count = 0
        self._snapshot: dict = {}
        self._terminated = False

        # Make sure resources are cleaned up if the user forgets to call close().
        atexit.register(self._safe_close)

        self._start_server()

    # ----- lifecycle ------------------------------------------------------

    def _start_server(self) -> None:
        env = os.environ.copy()
        env["WEBGYM_DB"] = str(self._db_path)
        env["WEBGYM_SEED"] = "0"
        # Run as a module so it picks up the project's `app` package on PYTHONPATH.
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
        self._server = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "app.asgi:app",
                "--host", "127.0.0.1",
                "--port", str(self._port),
                "--log-level", "warning",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _wait_for_http(self._base_url + "/", timeout_s=10.0)

    def _start_playwright(self) -> None:
        # Import lazily so unit-testing the verifiers doesn't require Playwright.
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context()
        self._page = self._context.new_page()

    # ----- public API -----------------------------------------------------

    def reset(self, seed: int = 0) -> tuple[dict, dict]:
        # 1. Reset the SQLite DB with the base seed.
        db.reset_database(self._db_path, seed)
        # 2. Apply task-specific seeding (e.g., insert a cancelable order).
        if self.task.seed_extra is not None:
            self.task.seed_extra(str(self._db_path), seed)
        # 3. Snapshot DB state for the verifier to diff against later.
        self._snapshot = SNAPSHOTS[self.task.id](str(self._db_path))
        self._step_count = 0
        self._terminated = False

        # 4. (Re-)open a clean browser context so cart/coupon cookies don't leak.
        if self._page is None:
            self._start_playwright()
        else:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = self._browser.new_context()
            self._page = self._context.new_page()

        self._page.goto(self._base_url + "/", wait_until="domcontentloaded")
        return self._observation(), {"task_id": self.task.id, "instruction": self.task.instruction}

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        if self._terminated:
            raise RuntimeError("step() called after termination; call reset() first")

        self._step_count += 1
        truncated = False
        info: dict[str, Any] = {"action": action}

        try:
            self._apply_action(action)
        except Exception as e:                                # noqa: BLE001
            info["action_error"] = repr(e)

        success, verify_info = self.task.verify(str(self._db_path), self._snapshot)
        info["verify"] = verify_info

        terminated = bool(success)
        reward = 1.0 if success else 0.0
        if self._step_count >= self.max_steps and not terminated:
            truncated = True

        self._terminated = terminated or truncated
        return self._observation(), reward, terminated, truncated, info

    def close(self) -> None:
        self._safe_close()

    # ----- introspection used by oracles ---------------------------------

    @property
    def page(self):
        """Direct Playwright page handle. Oracles use this to drive the UI
        without bouncing through the structured action space."""
        return self._page

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def db_path(self) -> str:
        return str(self._db_path)

    # ----- internals ------------------------------------------------------

    def _apply_action(self, action: dict) -> None:
        kind = action.get("type")
        page = self._page
        if kind == "noop":
            return
        if kind == "click":
            page.locator(action["selector"]).first.click()
        elif kind == "type":
            loc = page.locator(action["selector"]).first
            if action.get("clear", True):
                loc.fill(action.get("text", ""))
            else:
                loc.type(action.get("text", ""))
        elif kind == "scroll":
            dx = int(action.get("dx", 0))
            dy = int(action.get("dy", 0))
            page.evaluate(f"window.scrollBy({dx}, {dy})")
        elif kind == "navigate":
            target = action["url"]
            if target.startswith("/"):
                target = self._base_url + target
            page.goto(target, wait_until="domcontentloaded")
        else:
            raise ValueError(f"unknown action type: {kind!r}")

        # Most actions cause a navigation or form post; wait briefly so the
        # next observation reflects the post-action page rather than a
        # mid-flight state.
        try:
            page.wait_for_load_state("domcontentloaded", timeout=2000)
        except Exception:
            pass

    def _observation(self) -> dict:
        page = self._page
        obs: dict[str, Any] = {
            "url": page.url,
            "dom": page.content(),
        }
        try:
            obs["axtree"] = page.accessibility.snapshot() or {}
        except Exception:
            obs["axtree"] = {}
        if self.screenshot_enabled:
            try:
                obs["screenshot"] = page.screenshot(full_page=False)
            except Exception:
                obs["screenshot"] = b""
        else:
            obs["screenshot"] = b""
        return obs

    def _safe_close(self) -> None:
        # Idempotent. Tear down browser → server → workdir, in that order.
        try:
            if self._context is not None:
                self._context.close()
        except Exception:
            pass
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass
        self._context = self._browser = self._playwright = self._page = None

        if self._server is not None and self._server.poll() is None:
            try:
                self._server.send_signal(signal.SIGTERM)
                self._server.wait(timeout=3)
            except Exception:
                try:
                    self._server.kill()
                except Exception:
                    pass
            self._server = None

        if not self.keep_workdir and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
