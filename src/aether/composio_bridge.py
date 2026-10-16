"""Composio — the one connection hub for external apps.

Gmail, GitHub, Notion, Linear and friends connect through Composio, never
through per-app OAuth code of Aether's own. The shape:

- One session per Aether instance (sessions-first SDK): created with the
  direct-tools preset, so every capability is a real named tool on the
  session's hosted MCP endpoint — no meta-tools hiding actions inside
  params, which Aether's name-based policy could never see.
- That endpoint is registered once as the `composio` mcp_servers entry
  (SERVER_ENTRY), its URL and key referenced by $NAME and re-resolved from
  the encrypted store at every reconnect — so a regenerated session needs
  no config write and no restart. Tools arrive as composio__GMAIL_SEND_EMAIL
  and flow through the same Policy.classify → approval → audit choke point
  as everything else. This module never executes a tool.
- Connecting an app is a Composio-hosted Connect Link: credentials pass
  between the user and Composio only. Aether sees the toolkit's name and
  the connection's status, never a secret.
- Isolation is by a stable per-instance user id, generated once and kept
  in the secret store — never a shared "default" user.

The SDK is synchronous; every call rides asyncio.to_thread so the agent
loop never stalls on Composio's HTTP.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .secret_env import EnvResolver
    from .secret_store import SecretStore

log = logging.getLogger("aether.composio")

API_KEY_NAME = "COMPOSIO_API_KEY"
_USER_ID_NAME = "COMPOSIO_USER_ID"
_SESSION_ID_NAME = "COMPOSIO_SESSION_ID"
MCP_URL_NAME = "COMPOSIO_MCP_URL"
MCP_KEY_NAME = "COMPOSIO_MCP_KEY"

# the mcp_servers entry the /apps walk writes once — $NAME refs only, so the
# pinned config never carries a value and a regenerated session flows in
# through the resolver
SERVER_NAME = "composio"
SERVER_ENTRY: dict[str, Any] = {
    "name": SERVER_NAME,
    "transport": {
        "type": "http",
        "url": f"${MCP_URL_NAME}",
        "headers": {"x-api-key": f"${MCP_KEY_NAME}"},
    },
}

_CONNECT_TIMEOUT_SECONDS = 15 * 60.0  # a link nobody clicks stops waiting
_LIST_PAGE_LIMIT = 10  # pagination cap — a personal account never has 1000 apps
_TOOLKIT_CACHE_SECONDS = 3600.0  # the catalog barely moves; the panel reloads often


class ComposioNotConfigured(Exception):
    """No COMPOSIO_API_KEY anywhere the resolver looks."""


class ComposioUnavailable(Exception):
    """Configured, but Composio itself didn't answer (or answered badly)."""


@dataclass(frozen=True)
class ConnectedApp:
    """One connected account, panel-shaped. `id` is Composio's nanoid —
    not a credential, but never logged beyond debug."""

    id: str
    toolkit: str  # "gmail", "github", ...
    status: str  # ACTIVE / INITIATED / EXPIRED / ...
    identity: str  # best-effort account label (email, username…), "" when unknown


@dataclass(frozen=True)
class ToolkitInfo:
    """One app Composio can connect — the panel's grid and the walk's
    search both read this shape."""

    slug: str  # "gmail" — the authorize() argument
    name: str  # "Gmail"
    logo: str  # icon URL, "" when Composio has none
    description: str  # one line, may be ""


class ComposioBridge:
    """The narrow seam between Aether and the Composio SDK. Always
    constructed (main.py wires it unconditionally); every method is a no-op
    or an honest error until a COMPOSIO_API_KEY is resolvable — a key pasted
    in chat works without a restart because the key is resolved fresh."""

    def __init__(self, resolver: EnvResolver, secret_store: SecretStore) -> None:
        self._resolver = resolver
        self._secrets = secret_store
        self._sdk: Any = None  # composio.Composio, built lazily (sync SDK)
        self._session: Any = None  # the instance's one session, once opened
        self._user_id = ""
        self._lock = asyncio.Lock()
        self._toolkits_cache: tuple[float, list[ToolkitInfo]] | None = None

    async def available(self) -> bool:
        """A key exists — the walk's 'is Composio set up yet?'"""
        return bool(await self._resolver.resolve(API_KEY_NAME))

    async def ensure(self) -> bool:
        """Open (or reopen) the instance's session and persist its MCP
        coordinates. Idempotent; False when no key is set anywhere."""
        async with self._lock:
            key = await self._resolver.resolve(API_KEY_NAME)
            if not key:
                return False
            if self._session is not None:
                return True
            try:
                if self._sdk is None:
                    from composio import Composio

                    self._sdk = Composio(api_key=key)
                self._user_id = await self._instance_user_id()
                self._session = await self._open_session()
            except ComposioNotConfigured:
                raise
            except Exception as exc:
                raise ComposioUnavailable(str(exc)) from exc
            return True

    async def reset_session(self) -> bool:
        """Drop the cached session and cut a fresh one — new MCP URL and
        key, persisted over the old $NAME secrets, so the host's next
        reconnect re-resolves onto the new endpoint. This is the escape
        hatch the reuse path can't take: a router that accepts initialize
        but wedges on tools/list still answers sessions.use fine, so
        ensure() would ride the dead session forever. False when no key
        is set."""
        async with self._lock:
            key = await self._resolver.resolve(API_KEY_NAME)
            if not key:
                return False
            if self._sdk is None:
                from composio import Composio

                self._sdk = Composio(api_key=key)
            if not self._user_id:
                self._user_id = await self._instance_user_id()
            try:
                self._session = await self._open_session(force_new=True)
            except Exception as exc:
                raise ComposioUnavailable(str(exc)) from exc
            return True

    async def accounts(self) -> list[ConnectedApp]:
        """The instance's connected accounts — [] when Composio isn't
        configured (an honest empty list, not an error)."""
        if not await self.ensure():
            return []
        items = await asyncio.to_thread(self._list_accounts)
        return [self._to_app(item) for item in items]

    async def authorize(self, toolkit: str) -> tuple[str, str]:
        """Start connecting one toolkit → (request id, Connect Link). The
        link is the whole sign-in: credentials pass between the user and
        Composio only."""
        if not await self.ensure():
            raise ComposioNotConfigured(f"no {API_KEY_NAME} set")
        await self._sync_toolkits(extra=toolkit)
        try:
            request = await asyncio.to_thread(self._session.authorize, toolkit)
            if not request.redirect_url:
                raise ComposioUnavailable(f"composio gave no connect link for {toolkit}")
        except Exception:
            # the toolkit was only pre-admitted for this attempt — authorize
            # failed or gave no link, so drop it back out of the allowlist
            # rather than leave the session overclaiming a tool with no account
            try:
                await self._sync_toolkits()
            except Exception:
                log.warning("composio rollback sync failed", exc_info=True)
            raise
        return request.id, request.redirect_url

    async def wait_and_enable(
        self, toolkit: str, request_id: str, timeout: float = _CONNECT_TIMEOUT_SECONDS
    ) -> ConnectedApp:
        """Block (in a thread) until the Connect Link completes, then sync
        the session's toolkit allowlist so the hosted MCP starts listing
        the app's tools — the host's re-list picks them up within a minute.
        Raises on timeout or a terminally failed connection."""
        if not await self.ensure():
            raise ComposioNotConfigured(f"no {API_KEY_NAME} set")
        from composio.core.models.connected_accounts import ConnectionRequest

        request = await asyncio.to_thread(ConnectionRequest.from_id, request_id, self._sdk.client)
        try:
            connection = await asyncio.to_thread(request.wait_for_connection, timeout)
        except Exception:
            # the link was abandoned or failed — drop the toolkit
            # authorize() pre-admitted, or the session keeps listing tools
            # for an app that never connected
            await self._rollback_toolkits()
            raise
        await self._sync_toolkits()
        return self._to_app(connection)

    async def toolkits(self) -> list[ToolkitInfo]:
        """Every app Composio can connect for this project — the live
        catalog, not a list Aether maintains. Cached for an hour: the panel
        and the walk read it constantly, Composio's side barely moves. []
        when unconfigured."""
        if self._toolkits_cache is not None:
            at, cached = self._toolkits_cache
            if time.monotonic() - at < _TOOLKIT_CACHE_SECONDS:
                return cached
        if not await self.ensure():
            return []
        infos = await asyncio.to_thread(self._list_toolkits)
        self._toolkits_cache = (time.monotonic(), infos)
        return infos

    async def find_toolkits(self, query: str, limit: int = 5) -> list[ToolkitInfo]:
        """`/apps add gmail` → the toolkit it means. An exact slug or name
        match wins outright; otherwise the closest few by fuzzy score."""
        catalog = await self.toolkits()
        needle = query.strip().lower()
        if not needle:
            return []
        exact = [t for t in catalog if needle in (t.slug.lower(), t.name.lower())]
        if exact:
            return exact[:1]
        from rapidfuzz import fuzz

        scored = sorted(
            (
                (
                    max(
                        fuzz.ratio(needle, t.slug.lower()),
                        fuzz.partial_ratio(needle, t.name.lower()),
                    ),
                    t,
                )
                for t in catalog
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [t for score, t in scored[:limit] if score >= 50]

    def _list_toolkits(self) -> list[ToolkitInfo]:
        """The connectable catalog (sync; runs in a thread): not
        deprecated, and either composio-managed auth (the Connect Link just
        works) or no auth at all — anything else would need credentials
        set up outside the link flow, which the walk can't promise."""
        out: list[ToolkitInfo] = []
        cursor = None
        for _ in range(_LIST_PAGE_LIMIT * 2):  # the full catalog is bigger than one account's apps
            kwargs: dict[str, Any] = {"limit": 100, "sort_by": "alphabetically"}
            if cursor:
                kwargs["cursor"] = cursor
            page = self._sdk.client.toolkits.list(**kwargs)
            for item in page.items:
                # note: the list response's `deprecated` is a legacy id-mapping
                # object present on every toolkit, not a flag — nothing here
                # marks a toolkit as deprecated, so nothing is dropped for it
                if not item.no_auth and not item.composio_managed_auth_schemes:
                    continue
                meta = item.meta
                out.append(
                    ToolkitInfo(
                        slug=item.slug,
                        name=item.name,
                        logo=(meta.logo if meta else "") or "",
                        description=(meta.description if meta else "") or "",
                    )
                )
            cursor = page.next_cursor
            if not cursor:
                break
        out.sort(key=lambda t: t.name.lower())
        return out

    async def disconnect(self, account_id: str) -> ConnectedApp | None:
        """Remove one connected account — the panel hands over an arbitrary
        id, so ownership is verified before anything is deleted. None when
        the id isn't this instance's account."""
        if not await self.ensure():
            raise ComposioNotConfigured(f"no {API_KEY_NAME} set")
        from composio_client import NotFoundError

        try:
            account = await asyncio.to_thread(
                self._sdk.client.connected_accounts.retrieve, nanoid=account_id
            )
        except NotFoundError:
            return None
        if account.user_id != self._user_id:
            return None
        await asyncio.to_thread(
            self._sdk.client.connected_accounts.delete, account_id, revoke_on_delete=True
        )
        await self._sync_toolkits()
        return self._to_app(account)

    # -- internals ---------------------------------------------------------------

    async def _instance_user_id(self) -> str:
        """One stable id per Aether instance, generated once — Composio
        scopes connected accounts to it, so another Aether (or anyone else)
        can never see or use them."""
        stored = await self._secrets.get(_USER_ID_NAME)
        if stored:
            return stored
        fresh = f"aether-{uuid.uuid4().hex[:16]}"
        await self._secrets.set(_USER_ID_NAME, fresh)
        return fresh

    async def _open_session(self, force_new: bool = False) -> Any:
        """Reuse the stored session when it still exists server-side;
        otherwise create a fresh one carrying whatever is still connected,
        so a lost session never loses the apps. `force_new` skips the
        reuse entirely — reset_session's answer to a session that exists
        but whose endpoint no longer answers."""
        from composio import SESSION_PRESET_DIRECT_TOOLS

        session_id = None if force_new else await self._secrets.get(_SESSION_ID_NAME)
        if session_id:
            try:
                session = await asyncio.to_thread(self._sdk.sessions.use, session_id, mcp=True)
            except Exception:
                log.info("stored composio session is gone — creating a fresh one")
            else:
                await self._persist_mcp(session)
                return session
        toolkits = sorted(
            {app.toolkit for app in await self._list_apps() if app.status == "ACTIVE"}
        )
        session = await asyncio.to_thread(
            lambda: self._sdk.sessions.create(
                user_id=self._user_id,
                toolkits=toolkits,
                session_preset=SESSION_PRESET_DIRECT_TOOLS,
                mcp=True,
                # the preset preloads "all", which the API rejects against an empty
                # allowlist — the first-ever session has nothing connected yet, so
                # it starts silent; the first connect's _sync_toolkits fills it in
                **({"preload": {"tools": []}} if not toolkits else {}),
            )
        )
        await self._persist_mcp(session)
        return session

    async def _persist_mcp(self, session: Any) -> None:
        """The session's hosted-MCP coordinates into the encrypted store —
        the composio mcp_servers entry references them by $NAME, so the
        host picks up a regenerated session at its next reconnect."""
        headers = session.mcp.headers or {}
        key = headers.get("x-api-key") or next((v for v in headers.values() if v), "")
        if not session.mcp.url or not key:
            raise ComposioUnavailable("composio session gave no usable MCP endpoint")
        await self._secrets.set(_SESSION_ID_NAME, session.session_id)
        await self._secrets.set(MCP_URL_NAME, session.mcp.url)
        await self._secrets.set(MCP_KEY_NAME, key)

    async def _rollback_toolkits(self) -> None:
        """Drop any un-connected toolkits pre-admitted into the session allowlist."""
        try:
            await self._sync_toolkits()
        except Exception:
            log.warning("composio rollback sync failed", exc_info=True)

    async def _sync_toolkits(self, extra: str | None = None) -> None:
        """The session's allowlist mirrors what's connected — connect grows
        it, disconnect shrinks it, and the host's re-list makes the
        namespace follow. `extra` lets a pending connect in: the session
        refuses to even *authorize* a toolkit outside its allowlist, so the
        link step needs the app let in before any account is ACTIVE."""
        toolkits = sorted(
            {app.toolkit for app in await self._list_apps() if app.status == "ACTIVE"}
            | ({extra} if extra else set())
        )
        await asyncio.to_thread(
            self._session.update,
            toolkits={"enable": toolkits},
            # "all" is only legal with a positive allowlist — match the preload
            # to the allowlist so the MCP listing follows what's connected
            preload={"tools": "all" if toolkits else []},
        )

    async def _list_apps(self) -> list[ConnectedApp]:
        items = await asyncio.to_thread(self._list_accounts)
        return [self._to_app(item) for item in items]

    def _list_accounts(self) -> list[Any]:
        """All of this instance's connected accounts (sync; runs in a
        thread). Scoped by user id — the isolation boundary."""
        out: list[Any] = []
        cursor = None
        for _ in range(_LIST_PAGE_LIMIT):
            kwargs: dict[str, Any] = {"user_ids": [self._user_id], "limit": 100}
            if cursor:
                kwargs["cursor"] = cursor
            page = self._sdk.client.connected_accounts.list(**kwargs)
            out.extend(page.items)
            cursor = page.next_cursor
            if not cursor:
                break
        return out

    @staticmethod
    def _to_app(item: Any) -> ConnectedApp:
        """One account row → panel shape. The identity is best-effort:
        Composio exposes different account fields per toolkit, so the
        common ones are tried in order."""
        data = item.data or {}
        identity = ""
        for key in ("email", "user_email", "username", "login", "name", "display_name"):
            value = data.get(key)
            if isinstance(value, str) and value:
                identity = value
                break
        if not identity:
            identity = getattr(item, "alias", None) or ""
        return ConnectedApp(
            id=item.id,
            toolkit=item.toolkit.slug,
            status=str(item.status),
            identity=identity,
        )
