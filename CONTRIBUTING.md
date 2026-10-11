# Contributing to Aether

Thanks for helping build a personal agent people can actually trust.

## Getting set up

```bash
git clone <your-fork>
cd aether
cp .env.example .env        # you only need real values for a live smoke test
cp config.example.yaml config.yaml
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest                                  # unit tests — no network, no database
AETHER_TEST_DATABASE_URL=<url> pytest -m integration   # opt-in, needs Postgres

ruff check .                            # lint
ruff format --check .                   # formatting

playwright install chromium             # one-time, for the browser tests below
AETHER_TEST_E2E=1 pytest -m e2e          # opt-in, no database or API keys needed
```

CI runs this same gauntlet on every pull request — ruff check and format, the
unit suite on Python 3.12 and 3.14, the integration suite against a
containerized Postgres, the webapp build, and the e2e suite under Chromium
(`.github/workflows/ci.yml`). A PR is only done when CI is green.

Unit tests must stay hermetic: no network calls, no database, no API keys.
Anything touching a real provider or Postgres goes behind the `integration`
marker. `tests/fakes.py` holds the in-memory provider/connector doubles —
extend those rather than mocking SDKs ad hoc.

The `e2e` suite (`tests/e2e/`) drives the real frontend with a real browser
against `tests/e2e/dev_server.py` — the real FastAPI app wired to in-memory
fakes instead of Postgres. The panel source lives in `webapp/` (React +
Tailwind, built by Vite into `src/aether/web/`); build it first with
`cd webapp && npm install && npm run build`, or the e2e tests skip. Extend
that fixture and `tests/e2e/test_smoke.py` when you change frontend behavior;
don't verify frontend changes by manual screenshotting alone.

For hands-on frontend work, `npm run dev` serves the panel with `/api`
proxied to a locally running `aether` (port 8000). To click through against
the fixture instead, start `python tests/e2e/dev_server.py` (token
`secret`) and run `AETHER_DEV_API=http://127.0.0.1:8731 npm run dev:test` —
it skips unless `AETHER_DEV_API` is set, so a test run never silently
targets the real backend.

## Code conventions

- Python 3.12+, async everywhere — the whole service is one asyncio process.
- `ruff check` and `ruff format --check` must pass; both are configured in
  `pyproject.toml`.
- Plain SQL through asyncpg; no ORM. Migrations are forward-only files in
  `src/aether/migrations/`.
- Anything a human would consider private gets encrypted before it touches
  the database (`aether.memory.crypto`) and is bound to its record via AAD.
- The authorization layer stays **deterministic**: no LLM in the decision
  path, only code and declared rules.
- Every connector failure degrades gracefully — one broken MCP server or bot
  token must never take the agent loop down.

## Reporting issues

Open an issue with: what you did, what you expected, what happened, and the
relevant entries from the audit log (`/api/audit`).
