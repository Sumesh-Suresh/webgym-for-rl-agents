from __future__ import annotations

import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote

import gymnasium as gymnasium
import uvicorn
from gymnasium import spaces
from playwright.sync_api import Page, sync_playwright

from app import db
from app.main import create_app
from gym.actions import Action, Click, Navigate, Scroll, TypeText, coerce_action, make_action_space
from gym.task import AbstractEcommerceTask
from gym.ui_actions import UiAction, available_actions


class EcommerceEnv(gymnasium.Env):
    """Gymnasium-style browser environment for the local FastAPI storefront."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 1}

    def __init__(
        self,
        *,
        seed: int | None = None,
        headless: bool = True,
        timeout_ms: int = 2_000,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.task: AbstractEcommerceTask | None = None
        self.seed_value = seed
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.debug = debug
        self.action_space = make_action_space()
        self.observation_space = spaces.Dict(
            {
                "url": spaces.Text(max_length=2_048),
                "dom": spaces.Text(max_length=250_000),
                "screenshot": spaces.Sequence(spaces.Box(low=0, high=255, shape=(), dtype=int)),
            }
        )
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.db_path: Path | None = None
        self.base_url: str | None = None
        self._server: uvicorn.Server | None = None
        self._server_thread: threading.Thread | None = None
        self._playwright = None
        self._browser = None
        self._context = None
        self.page: Page | None = None
        self._observation_index = 0
        self.last_action: Action | None = None
        self.last_action_error: str = ""

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.close()
        self._observation_index = 0
        actual_seed = self.seed_value if seed is None else seed
        self._tmpdir = tempfile.TemporaryDirectory(prefix="webgym-ecommerce-")
        self.db_path = Path(self._tmpdir.name) / "store.sqlite"
        db.reset_database(self.db_path, actual_seed)
        self.task = AbstractEcommerceTask.sample(actual_seed, self.db_path)
        self._start_server(actual_seed)
        self._start_browser()
        assert self.page is not None
        assert self.task is not None
        self.page.goto(f"{self.base_url}/", wait_until="networkidle")
        return self._observation(), {
            "instruction": self.task.instruction,
            "task_kind": self.task.kind,
            "task_name": self.task.name,
            "db_path": str(self.db_path),
            "available_actions": available_actions(self.page.url),
        }

    def step(
        self, action: Action | UiAction | tuple[int, dict[str, Any]] | dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.page is None or self.db_path is None or self.task is None:
            raise RuntimeError("call reset() before step()")

        parsed_action = coerce_action(action)
        self.last_action = parsed_action
        info: dict[str, Any] = {"action_exec_start": time.time()}
        try:
            self._execute_action(parsed_action)
            self.last_action_error = ""
        except Exception as exc:
            self.last_action_error = f"{type(exc).__name__}: {exc}"

        return self.post_step(info)

    def post_step(
        self, info: dict[str, Any], validate: bool = True
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        assert self.page is not None
        assert self.db_path is not None
        assert self.task is not None

        info["action_exec_stop"] = time.time()
        self._wait_dom_loaded()
        info["validation_start"] = time.time()

        if validate:
            reward, done, user_message, task_info = self._task_validate()
            info["task_info"] = task_info
        else:
            reward, done, user_message = 0.0, False, ""
            info["task_info"] = {}

        info["validation_stop"] = time.time()
        if user_message:
            info["user_message"] = user_message

        info["get_observation_start"] = time.time()
        obs = self._observation()
        info["get_observation_stop"] = time.time()
        info["available_actions"] = available_actions(self.page.url)
        info["last_action_error"] = self.last_action_error

        return obs, reward, done, False, info

    def _execute_action(self, action: Action) -> None:
        assert self.page is not None
        assert self.base_url is not None

        if isinstance(action, Click):
            self.page.click(action.selector, timeout=self.timeout_ms)
        elif isinstance(action, TypeText):
            # page.fill() clears the field and sets the value in one call (page.type is deprecated).
            self.page.fill(action.selector, action.text)
        elif isinstance(action, Scroll):
            self.page.mouse.wheel(0, float(action.delta_y))
        elif isinstance(action, Navigate):
            url = action.url
            if url.startswith("/"):
                url = f"{self.base_url}{url}"
            # goto already waits for networkidle; return early to skip the trailing wait.
            self.page.goto(url, wait_until="networkidle", timeout=self.timeout_ms)
            return
        else:
            raise ValueError(f"unsupported action: {action!r}")
        self.page.wait_for_load_state("networkidle", timeout=self.timeout_ms)

    def _task_validate(self) -> tuple[float, bool, str, dict[str, Any]]:
        assert self.page is not None
        assert self.db_path is not None
        assert self.task is not None
        return self.task.validate(self.page, self.db_path)

    def _wait_dom_loaded(self) -> None:
        assert self.page is not None
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=self.timeout_ms)
        except Exception:
            pass

    def render(self):
        if self.page is None:
            return None
        return self.page.screenshot(type="png")

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
        if self._server is not None:
            self._server.should_exit = True
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=2)
            self._server_thread = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
        self.page = None
        self.base_url = None
        self.db_path = None
        self.task = None
        self._observation_index = 0
        self.last_action = None
        self.last_action_error = ""

    def _start_server(self, seed: int | None) -> None:
        assert self.db_path is not None
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("127.0.0.1", 0))
        server_socket.listen(128)
        port = server_socket.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"

        config = uvicorn.Config(
            create_app(self.db_path, seed),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._server_thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [server_socket]},
            daemon=True,
        )
        self._server_thread.start()
        self._wait_for_server()

    def _wait_for_server(self, timeout: float = 10.0) -> None:
        """Poll the server's root URL until it responds or the timeout expires."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(f"{self.base_url}/", timeout=1.0)
                return
            except (urllib.error.URLError, OSError):
                time.sleep(0.05)
        raise RuntimeError(f"server at {self.base_url} did not start within {timeout}s")

    def _start_browser(self) -> None:
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(viewport={"width": 1280, "height": 900})
        self.page = self._context.new_page()
        self.page.set_default_timeout(self.timeout_ms)

    def _observation(self) -> dict[str, Any]:
        assert self.page is not None
        assert self.task is not None

        dom = self.page.content()
        screenshot = self.page.screenshot(type="png")
        self._observation_index += 1

        if self.debug:
            observations_dir = Path("observations") / self.task.name
            observations_dir.mkdir(parents=True, exist_ok=True)
            url_slug = quote(self.page.url, safe="")
            filename = f"{self._observation_index:03d}-{url_slug}"
            (observations_dir / f"{filename}.html").write_text(dom)
            (observations_dir / f"{filename}.png").write_bytes(screenshot)

        return {
            "url": self.page.url,
            "dom": dom,
            "screenshot": screenshot,
        }
