"""Shared fixtures.

Unit tests stay hermetic (no network, no database). Integration tests are
opt-in: they run only when AETHER_TEST_DATABASE_URL is set, and manage a
migrated, truncated database themselves. E2E tests are opt-in the same way,
via AETHER_TEST_E2E=1 — they need `playwright install chromium` done once
first (see tests/e2e/README.md).
"""

from __future__ import annotations

import os

import pytest

from aether.memory.db import create_pool, run_migrations

DB_URL = os.environ.get("AETHER_TEST_DATABASE_URL", "")
E2E_ENABLED = os.environ.get("AETHER_TEST_E2E", "") == "1"

# Tables integration tests are allowed to wipe between tests.
_TABLES = (
    "audit_log, pending_approvals, events, identities, identity_handles, "
    "entity_notes, chat_messages, scheduled_actions, routines, decision_traces, "
    "config_overrides, oauth_tokens, app_secrets, memory_chunks"
)


def pytest_collection_modifyitems(config, items):
    skip_integration = pytest.mark.skip(reason="integration: set AETHER_TEST_DATABASE_URL to run")
    skip_e2e = pytest.mark.skip(
        reason="e2e: set AETHER_TEST_E2E=1 to run (needs `playwright install chromium` first)"
    )
    for item in items:
        if not DB_URL and "integration" in item.keywords:
            item.add_marker(skip_integration)
        if not E2E_ENABLED and "e2e" in item.keywords:
            item.add_marker(skip_e2e)
    # run e2e last: playwright's sync dispatcher, once started, keeps pytest-asyncio
    # auto mode from claiming any async test that runs after it in the same process
    items.sort(key=lambda i: "e2e" in i.keywords)


@pytest.fixture
async def db():
    """A migrated, freshly-truncated pool. Requires AETHER_TEST_DATABASE_URL."""
    pool = await create_pool(DB_URL)
    await run_migrations(pool)
    await pool.execute(f"TRUNCATE {_TABLES} RESTART IDENTITY")
    yield pool
    await pool.execute(f"TRUNCATE {_TABLES} RESTART IDENTITY")
    await pool.close()
