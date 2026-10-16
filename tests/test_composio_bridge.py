"""ComposioBridge unit tests — session lifecycle, allowlist sync, and rollback.

Hermetic unit tests: no network, no database, no real Composio API keys.
Uses in-memory doubles from tests/fakes.py.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

try:
    from fakes import FakeComposioSession, FakeSecretStore
except ModuleNotFoundError:
    # When running in environments where backend dependencies (e.g. asyncpg)
    # aren't installed, provide local doubles with the same surface.
    class FakeSecretStore:  # type: ignore[no-redef]
        def __init__(self, secrets: dict[str, str] | None = None) -> None:
            self.secrets = dict(secrets or {})

        async def get(self, name: str) -> str | None:
            return self.secrets.get(name)

        async def set(self, name: str, value: str) -> None:
            self.secrets[name] = value

        async def names(self) -> set[str]:
            return set(self.secrets)

    class FakeComposioSession:  # type: ignore[no-redef]
        def __init__(
            self,
            *,
            authorize_result: object | None = None,
            authorize_error: Exception | None = None,
            update_error: Exception | None = None,
            session_id: str = "fake-session-id",
            mcp_url: str = "https://mcp.composio.dev/fake",
            mcp_key: str = "fake-mcp-key",
        ) -> None:
            self.session_id = session_id
            self.mcp = type("MCP", (), {"url": mcp_url, "headers": {"x-api-key": mcp_key}})()
            self.authorize_result = authorize_result
            self.authorize_error = authorize_error
            self.update_error = update_error
            self.update_calls: list[dict[str, Any]] = []
            self.authorize_calls: list[str] = []

        def update(self, **kwargs: Any) -> None:
            self.update_calls.append(kwargs)
            if self.update_error is not None:
                raise self.update_error

        def authorize(self, toolkit: str) -> Any:
            self.authorize_calls.append(toolkit)
            if self.authorize_error is not None:
                raise self.authorize_error
            if self.authorize_result is not None:
                return self.authorize_result
            return type(
                "Request",
                (),
                {"id": f"req-{toolkit}", "redirect_url": f"https://hub.example.test/{toolkit}"},
            )()


from aether.composio_bridge import (
    API_KEY_NAME,
    ComposioBridge,
    ComposioNotConfigured,
    ComposioUnavailable,
    ConnectedApp,
)


class FakeEnvResolver:
    """EnvResolver double returning pre-set keys."""

    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})

    async def resolve(self, name: str) -> str:
        return self.values.get(name, "")


def _bridge(
    *,
    key: str = "fake-composio-key",
    session: FakeComposioSession | None = None,
    active_apps: list[ConnectedApp] | None = None,
) -> tuple[ComposioBridge, FakeComposioSession]:
    secrets = FakeSecretStore()
    resolver = FakeEnvResolver({API_KEY_NAME: key} if key else {})
    bridge = ComposioBridge(resolver, secrets)
    session = session or FakeComposioSession()
    bridge._session = session
    bridge._user_id = "test-user-id"

    # In-memory account listing double
    apps = list(active_apps or [])

    async def _fake_list_apps() -> list[ConnectedApp]:
        return list(apps)

    bridge._list_apps = _fake_list_apps  # type: ignore[assignment]
    return bridge, session


async def test_authorize_requires_configured_key() -> None:
    bridge, _ = _bridge(key="")
    with pytest.raises(ComposioNotConfigured):
        await bridge.authorize("gmail")


async def test_authorize_success() -> None:
    bridge, session = _bridge()
    req_id, redirect_url = await bridge.authorize("gmail")

    assert req_id == "req-gmail"
    assert redirect_url == "https://hub.example.test/gmail"
    assert session.authorize_calls == ["gmail"]
    # Exactly one allowlist update enabling gmail was performed (no rollback)
    assert session.update_calls == [
        {"toolkits": {"enable": ["gmail"]}, "preload": {"tools": "all"}}
    ]


async def test_authorize_rolls_back_on_session_authorize_exception() -> None:
    session = FakeComposioSession(authorize_error=RuntimeError("composio api error"))
    bridge, _ = _bridge(session=session)

    with pytest.raises(RuntimeError, match="composio api error"):
        await bridge.authorize("gmail")

    assert session.authorize_calls == ["gmail"]
    # Pre-admitted gmail, then rolled back on exception
    assert session.update_calls == [
        {"toolkits": {"enable": ["gmail"]}, "preload": {"tools": "all"}},
        {"toolkits": {"enable": []}, "preload": {"tools": []}},
    ]


async def test_authorize_rolls_back_on_missing_redirect_url() -> None:
    empty_req = type("Req", (), {"id": "req-empty", "redirect_url": ""})()
    session = FakeComposioSession(authorize_result=empty_req)
    bridge, _ = _bridge(session=session)

    with pytest.raises(ComposioUnavailable, match="no connect link"):
        await bridge.authorize("gmail")

    assert session.authorize_calls == ["gmail"]
    # Pre-admitted gmail, then rolled back when redirect_url was missing
    assert session.update_calls == [
        {"toolkits": {"enable": ["gmail"]}, "preload": {"tools": "all"}},
        {"toolkits": {"enable": []}, "preload": {"tools": []}},
    ]


async def test_authorize_rollback_failure_does_not_mask_original_exception() -> None:
    # authorize raises primary error, and rollback update raises secondary error
    session = FakeComposioSession(
        authorize_error=RuntimeError("primary auth error"),
    )
    bridge, _ = _bridge(session=session)

    # Make the second update() call (during rollback) raise
    first_call = True

    def flaky_update(**kwargs: Any) -> None:
        nonlocal first_call
        session.update_calls.append(kwargs)
        if not first_call:
            raise RuntimeError("rollback update failed")
        first_call = False

    session.update = flaky_update  # type: ignore[assignment]

    # The original RuntimeError("primary auth error") must still be raised
    with pytest.raises(RuntimeError, match="primary auth error"):
        await bridge.authorize("gmail")

    # Both update calls were attempted
    assert len(session.update_calls) == 2


async def test_rollback_preserves_already_active_apps() -> None:
    active_github = ConnectedApp(
        id="acc-gh",
        toolkit="github",
        status="ACTIVE",
        identity="user@example.test",
    )
    session = FakeComposioSession(authorize_error=RuntimeError("connect start rejected"))
    bridge, _ = _bridge(session=session, active_apps=[active_github])

    with pytest.raises(RuntimeError, match="connect start rejected"):
        await bridge.authorize("gmail")

    # Pre-admit added gmail to github; rollback restored only active apps (github)
    assert session.update_calls == [
        {"toolkits": {"enable": ["github", "gmail"]}, "preload": {"tools": "all"}},
        {"toolkits": {"enable": ["github"]}, "preload": {"tools": "all"}},
    ]


async def test_wait_and_enable_uses_rollback() -> None:
    session = FakeComposioSession()
    bridge, _ = _bridge(session=session)

    class FakeRequest:
        def wait_for_connection(self, timeout: float) -> None:
            raise TimeoutError("link click timed out")

    class FakeRequestClass:
        @classmethod
        def from_id(cls, req_id: str, client: Any) -> FakeRequest:
            return FakeRequest()

    import sys
    from types import ModuleType

    if "composio.core.models.connected_accounts" not in sys.modules:
        sys.modules.setdefault("composio", ModuleType("composio"))
        sys.modules.setdefault("composio.core", ModuleType("composio.core"))
        sys.modules.setdefault("composio.core.models", ModuleType("composio.core.models"))
        sys.modules["composio.core.models.connected_accounts"] = ModuleType(
            "composio.core.models.connected_accounts"
        )

    bridge._sdk = type("SDK", (), {"client": None})()
    with (
        patch(
            "composio.core.models.connected_accounts.ConnectionRequest",
            FakeRequestClass,
            create=True,
        ),
        pytest.raises(TimeoutError, match="link click timed out"),
    ):
        await bridge.wait_and_enable("gmail", "req-123")

    # Rollback sync was called after timeout
    assert session.update_calls == [
        {"toolkits": {"enable": []}, "preload": {"tools": []}},
    ]


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            print(f"Running {name}...")
            asyncio.run(func())
    print("All tests passed!")
