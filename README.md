# Minimal Web Gym for RL Agents

This repository contains a small deterministic e-commerce storefront and a
Gymnasium-style browser environment for training or evaluating web agents. The
web app is implemented with FastAPI and SQLite. The gym environment launches the
app locally, drives it with Playwright, and verifies task success against the
database.

## Requirements

- Python 3.10 or newer.
- `pip` and a virtual environment.
- Chromium browser binaries installed through Playwright if you use the gym.
- Docker is optional for running only the storefront.

Install the application dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r app/requirements-app.txt
```

Install the gym dependencies:

```bash
pip install -r gym/requirements-gym.txt
python -m playwright install chromium
```

## Launch the Web App Locally

From the repository root:

```bash
source .venv/bin/activate
WEBGYM_SEED=0 WEBGYM_DB=data/store.sqlite uvicorn app.asgi:app --reload --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/`.

`WEBGYM_SEED` controls the deterministic product/order seed. `WEBGYM_DB`
controls where the SQLite database is written. `app/asgi.py` resets the database
when the app starts, then exposes the FastAPI app.

You can also run the storefront with Docker:

```bash
docker compose -f app/docker-compose.yml up --build
```

## Storefront Routes

The app serves HTML pages and form endpoints, not a JSON API.

- `GET /` - product listing. Optional query parameters:
  - `category`: `Electronics`, `Fitness`, `Home`, or `Office`
  - `sort`: `alpha`, `price_asc`, or `price_desc`
- `GET /product/{sku}` - product detail page with an add-to-cart form.
- `POST /cart/add` - add a product to the cart. Form fields: `sku`,
  `quantity`.
- `GET /cart` - cart page with line items, coupon form, and checkout link.
- `POST /cart/update` - update a cart line quantity. Form fields: `sku`,
  `quantity`.
- `POST /cart/remove` - remove a cart line. Form field: `sku`.
- `POST /cart/coupon` - apply a coupon. Form field: `coupon_code`. The seeded
  coupon is `SAVE10`.
- `GET /checkout` - checkout form. Redirects to `/cart` when the cart is empty.
- `POST /checkout` - create an order. Form field: `shipping_address`.
- `GET /orders/{order_id}/confirmation` - order confirmation page.
- `GET /orders` - order history page.
- `GET /orders/{order_id}` - order detail page.
- `POST /orders/{order_id}/cancel` - cancel a placed order.

Cart and coupon state are stored in HTTP-only cookies. Product, coupon, and
order state are stored in SQLite.

## What the `gym` Folder Does

The `gym` package wraps the storefront as a Gymnasium environment:

- `gym/env.py` defines `EcommerceEnv`, which starts a temporary SQLite database,
  launches a FastAPI server on a free local port, opens Chromium with Playwright,
  and returns browser observations.
- `gym/task.py` defines sampled tasks and success checks:
  `buy_cheapest_in_category`, `apply_coupon_with_quantity`, and
  `cancel_recent_order`.
- `gym/actions.py` defines low-level browser actions: click, type, scroll, and
  navigate. It also converts dicts and Gymnasium `OneOf` samples into action
  objects.
- `gym/ui_actions.py` defines higher-level storefront actions such as
  `filter_products`, `open_product`, `fill_field`, `click_ui`, `view_order`, and
  `cancel_order`.
- `gym/policies.py` contains the scripted oracle policy and a simple random
  policy.
- `gym/debug_report.py` formats per-episode debug output for rollouts.

Each environment reset creates isolated state, samples a task from the seeded
database, starts the local web app, and navigates to the home page. Each step
executes an action in the browser, validates task success, and returns:

```python
observation, reward, terminated, truncated, info = env.step(action)
```

The observation contains:

- `url`: current page URL.
- `dom`: current HTML.
- `screenshot`: PNG screenshot bytes.

The reward is `1.0` when the task verifier passes and `0.0` otherwise.
`terminated` is true when the task is complete.

## Using `EcommerceEnv`

Example:

```python
from gym.env import EcommerceEnv
from gym.policies import scripted_oracle_actions

env = EcommerceEnv(seed=123, headless=True)

try:
    observation, info = env.reset(seed=123)
    print(info["instruction"])
    print(info["available_actions"])

    for action in scripted_oracle_actions(env):
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

    print("success:", reward > 0)
finally:
    env.close()
```

You can also pass low-level action dictionaries:

```python
env.step({"type": "navigate", "url": "/"})
env.step({"type": "click", "selector": "[data-testid='checkout-link']"})
env.step({"type": "type", "selector": "[data-testid='shipping-address']", "text": "123 Main St"})
env.step({"type": "scroll", "delta_y": 500})
```

Important precursors:

- Install `gym/requirements-gym.txt`.
- Run `python -m playwright install chromium`.
- Always call `reset()` before `step()`.
- Always call `close()` when finished so the browser, server, and temporary
  database are cleaned up.
- Use `headless=False` when you want to watch the browser.

## Example Scripts

The `example/` folder contains three standalone scripts that show the main ways
to customize an episode. Each script accepts `--seed`; omitting it uses the
script's default seed.

- `example/example_oracle.py` runs `gym.runner.run_episode()` with the built-in
  oracle policy and the task sampled by the environment. Use this when you want
  the simplest end-to-end rollout with no customization.
- `example/example_custom_task.py` calls `env.reset()` to initialize the browser
  and database, defines and injects `BuyMostExpensiveInCategoryTask`, and lets
  the built-in oracle complete that custom task.
- `example/example_custom_task_and_policy.py` defines and injects
  `BuyMostExpensiveInCategoryTask`, then drives it with a custom policy that
  returns a task-compatible list of actions.

Run them from the repository root:

```bash
python example/example_oracle.py
python example/example_custom_task.py
python example/example_custom_task_and_policy.py --seed 7
```

The scripts print the selected task, instruction where applicable, per-step URLs
and rewards for manual step loops, and the final success result.

## `rollout.py`

`rollout.py` is a command-line rollout runner. It executes multiple episodes
concurrently with `ThreadPoolExecutor`, creates one `EcommerceEnv` per episode,
resets it with a deterministic seed, runs either the scripted oracle policy or a
random policy, and prints success rates grouped by task kind.

Run the oracle policy:

```bash
python rollout.py --instances 4 --episodes 6 --policy oracle --max-steps 12
```

Run a random baseline:

```bash
python rollout.py --instances 4 --episodes 20 --policy random --max-steps 12
```

Useful flags:

- `--instances` - number of concurrent worker threads/environments.
- `--episodes` - total episodes to run.
- `--policy` - `oracle` or `random`.
- `--max-steps` - maximum actions per episode.
- `--debug` - print structured per-episode traces and a rollout summary.
- `--continue-after-success` - keep stepping even after the verifier succeeds.

Episode seeds start at `10000` and increment by episode index, so repeated runs
with the same arguments are deterministic unless the app logic changes.
