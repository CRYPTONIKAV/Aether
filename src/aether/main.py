"""FastAPI app factory + startup wiring."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import restart
from .agent import AgentLoop, CaptureRequestBox, SurfaceFanout, register_native_tools
from .agent.settings import AgentSettings
from .agent.traces import Traces
from .api import router as api_router
from .authz.approvals import Approvals
from .authz.audit import AuditLog
from .authz.policy import Policy, PolicyError
from .chat import ChatHistory, ChatHub
from .chat.ws import router as chat_router
from .composio_bridge import ComposioBridge
from .config import AppConfig, Settings, load_config
from .config_store import ConfigManager, ConfigOverrides
from .connectors import build_messaging_connectors
from .connectors.attachments import AttachmentProcessor
from .connectors.mcp_host import MCPHost
from .connectors.registry import ToolRegistry
from .connectors.screenvision import ScreenVision
from .llm.base import ProviderError
from .llm.registry import ProviderRegistry
from .logging_setup import configure_logging
from .memory.context import ContextBuilder
from .memory.crypto import Cipher, CryptoError, generate_key_b64
from .memory.db import create_pool, run_migrations
from .memory.entities import Entities, LLMSamePersonJudge
from .memory.events import EventStore
from .memory.salience import LLMJudge, Salience
from .routines import Routines
from .scheduler import Scheduler, SchedulerWorker
from .secret_env import EnvResolver
from .secret_store import SecretStore
from .workspace import Workspace, WorkspaceWriteError

log = logging.getLogger("aether.main")

WEB_DIR = Path(__file__).parent / "web"


class StartupError(RuntimeError):
    """Fatal misconfiguration — boot refuses to continue."""


def _build_cipher(settings: Settings) -> Cipher:
    if settings.encryption_key:
        try:
            return Cipher.from_b64(settings.encryption_key)
        except CryptoError as exc:
            raise StartupError(f"AETHER_ENCRYPTION_KEY is invalid: {exc}") from exc
    if settings.dev_ephemeral_key:
        log.warning(
            "AETHER_ENCRYPTION_KEY not set — using an EPHEMERAL key. "
            "Encrypted data will be unreadable after a restart. "
            "Set AETHER_DEV_EPHEMERAL_KEY=0 and configure a real key for anything real."
        )
        return Cipher.from_b64(generate_key_b64())
    raise StartupError(
        "AETHER_ENCRYPTION_KEY is not set. Generate one with:\n"
        '  python -c "from aether.memory.crypto import generate_key_b64; print(generate_key_b64())"\n'
        "(or set AETHER_DEV_EPHEMERAL_KEY=1 for a throwaway dev instance)"
    )


def create_app(
    settings: Settings | None = None,
    config_path: str | None = None,
) -> FastAPI:
    configure_logging()
    settings = settings or Settings()
    config: AppConfig = load_config(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # --- fail-fast validation -----------------------------------------
        if not settings.database_url:
            raise StartupError(
                "DATABASE_URL is not set. Aether persists everything to Postgres — "
                "create a free Neon database and put its connection string in .env."
            )
        cipher = _build_cipher(settings)

        # --- database ------------------------------------------------------
        pool = await create_pool(settings.database_url)
        applied = await run_migrations(pool)
        log.info("database ready (migrations applied this boot: %s)", applied or "none")

        # --- secrets by name + the app hub ---------------------------------
        # keys pasted in the /apps walk (consumed before ingest) live
        # encrypted like everything else; the resolver is how every act
        # reads a key — by name, fresh — so a key added after boot needs no
        # restart. The Composio bridge rides it: a COMPOSIO_API_KEY pasted
        # in chat works without a boot.
        secret_store = SecretStore(pool, cipher)
        env_resolver = EnvResolver(secrets=secret_store)
        app.state.resolver = env_resolver
        composio = ComposioBridge(env_resolver, secret_store)
        app.state.composio = composio
        try:
            if await composio.ensure():
                log.info("composio session open — the app hub is live")
        except Exception:
            # the hub being down must never keep Aether from booting —
            # /apps answers honestly and the next connect retries
            log.exception("composio session couldn't open — apps connect on demand")

        # --- chat-made config: rows merge onto the live config, in place ----
        # Before anything reads a section — an authz or allowlist override
        # must already be in effect when the first Policy and store are built.
        config_manager = ConfigManager(ConfigOverrides(pool, cipher), settings, config)
        await config_manager.bootstrap()
        app.state.config_manager = config_manager

        try:
            policy = Policy(config.authz.rules)
        except PolicyError as exc:
            raise StartupError(f"bad authz rule: {exc}") from exc

        app.state.pool = pool
        app.state.cipher = cipher
        app.state.settings = settings
        app.state.config = config

        # --- memory + authz ------------------------------------------------
        audit = AuditLog(pool)
        events = EventStore(pool, cipher, audit, config.contacts)
        approvals = Approvals(pool, cipher, audit, config.authz.approval_ttl_hours)
        agent_settings = AgentSettings(pool, cipher)
        app.state.audit = audit
        app.state.events = events
        app.state.approvals = approvals
        app.state.agent_settings = agent_settings

        # --- connectors: MCP host + the flat tool namespace -----------------
        host = MCPHost(config.mcp_servers, resolver=env_resolver)
        host.start()
        app.state.mcp_host = host
        tools = ToolRegistry()
        tools.attach_mcp(host)
        log.info("mcp tools in the namespace: %d", tools.sync_mcp_tools())
        app.state.tools = tools

        # --- llm providers (best effort: the agent boots without keys) ----
        try:
            providers = ProviderRegistry.from_config(settings, config.llm)
        except ProviderError as exc:
            providers = None
            log.warning(
                "LLM provider unavailable (%s) — chat replies, salience judging, "
                "and screen vision stay off until the keys are configured",
                exc,
            )
        app.state.providers = providers
        app.state.screen_vision = ScreenVision(providers, events) if providers else None
        # shared chat files ride the same rails — the loop's drain calls it
        app.state.attachments = AttachmentProcessor(providers, events) if providers else None

        judge_provider = providers.for_role("salience") if providers else None
        entities = Entities(pool, cipher, audit, LLMSamePersonJudge(judge_provider))
        salience = Salience(events, LLMJudge(judge_provider), config.salience)
        context = ContextBuilder(events, entities, approvals, config.agent)
        scheduler = Scheduler(pool, cipher, audit)
        app.state.entities = entities
        app.state.scheduler = scheduler
        routines = Routines(pool, cipher, audit)
        app.state.routines = routines
        traces = Traces(pool, cipher)
        app.state.traces = traces

        # --- workspace: the human-editable Markdown identity/memory layer ---
        # an enhancement layer, not core plumbing: a workspace that can't be
        # set up (bad permissions, a full disk) degrades to "no workspace"
        # rather than taking the whole boot down with it.
        workspace: Workspace | None = None
        if config.workspace.enabled:
            candidate = Workspace(config.workspace.path)
            try:
                first_run = candidate.ensure_scaffold()
            except WorkspaceWriteError as exc:
                log.warning("workspace unavailable (%s) — continuing without it", exc)
            else:
                workspace = candidate
                log.info("workspace ready at %s (first run: %s)", workspace.root, first_run)
        app.state.workspace = workspace

        # --- chat + the agent ------------------------------------------------
        chat_history = ChatHistory(pool, cipher)
        chat_hub = ChatHub(chat_history)
        capture_box = CaptureRequestBox()
        app.state.chat_history = chat_history
        app.state.chat_hub = chat_hub
        app.state.capture_box = capture_box

        surfaces = SurfaceFanout(hub=chat_hub, history=chat_history)
        agent = AgentLoop(
            providers=providers,
            tools=tools,
            policy=policy,
            approvals=approvals,
            audit=audit,
            events=events,
            salience=salience,
            context=context,
            scheduler=scheduler,
            surfaces=surfaces,
            capture_box=capture_box,
            config=config,
            host=host,
            routines=routines,
            traces=traces,
            agent_settings=agent_settings,
            config_manager=config_manager,
            workspace=workspace,
            resolver=env_resolver,
            secret_store=secret_store,
            settings=settings,
            composio=composio,
            transcript=chat_history,
            attachments=app.state.attachments,
        )
        app.state.agent = agent
        native = register_native_tools(
            registry=tools,
            events=events,
            entities=entities,
            approvals=approvals,
            scheduler=scheduler,
            surfaces=surfaces,
            capture_box=capture_box,
            routines=routines,
            traces=traces,
            audit=audit,
            config_manager=config_manager,
            workspace=workspace,
        )
        log.info("native tools registered: %d", native)

        connectors = build_messaging_connectors(
            settings,
            config,
            approvals=approvals,
            on_decision=agent.execute_decision,
            on_inbound=agent.handle_inbound,
        )
        surfaces.add_connectors(connectors)
        for connector in connectors:
            await connector.start()

        # the manager can now live-apply a chat-made change: swap the policy,
        # diff the connectors, diff the mcp host, re-parse quiet hours, or
        # arm a self-restart for an llm.* change
        config_manager.wire(
            loop=agent,
            surfaces=surfaces,
            tools=tools,
            host=host,
            approvals=approvals,
            resolver=env_resolver,
        )

        worker = SchedulerWorker(scheduler, agent.execute_scheduled)
        agent.start()
        worker.start()
        log.info(
            "aether running — surfaces: web%s, agent loop + scheduler live",
            "".join(f", {c.name}" for c in connectors),
        )

        yield

        await worker.stop()
        await agent.stop()
        for connector in connectors:
            await connector.stop()
        await host.stop()
        await pool.close()
        log.info("aether stopped")

    app = FastAPI(title="Aether", version="0.1.0", lifespan=lifespan)
    app.include_router(api_router)
    app.include_router(chat_router)
    # the panel is a Vite build artifact (source lives in webapp/) — serve it
    # when present, and say so plainly when it isn't, so a fresh checkout
    # boots without the mount crashing on a directory it doesn't have
    panel_built = (WEB_DIR / "index.html").is_file()
    if panel_built:
        app.mount("/assets", StaticFiles(directory=WEB_DIR), name="web-assets")

    @app.get("/healthz")
    @app.get("/health")  # some pingers default here; same plain 200
    async def healthz() -> dict:
        """Plain 200 — this is the endpoint the external uptime pinger hits."""
        return {"ok": True}

    @app.get("/v1/models")
    async def list_models() -> dict:
        """The OpenAI-compatible models answer. Local capability probes ask
        this right after /health to check whether an OpenAI-style server
        lives on the port; before this route the honest answer was a 404,
        the only error line in an otherwise clean log. Model ids are not
        secrets — the same reasoning that leaves /health unguarded."""
        llm = config.llm
        ids = {llm.model} | {m for m in (llm.vision_model, llm.salience_model) if m}
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "created": 0, "owned_by": llm.provider}
                for m in sorted(ids)
            ],
        }

    @app.get("/")
    async def index() -> Response:
        if not panel_built:
            return PlainTextResponse(
                "panel not built — run: cd webapp && npm install && npm run build",
                status_code=503,
            )
        return FileResponse(WEB_DIR / "index.html")

    return app


def main() -> None:
    """Console entry point (`aether` / `uv run aether`). A serve loop, not a
    single run: a chat-made llm.* change arms restart.request(), which flips
    this server's should_exit once the current turn has had time to land its
    words; the loop then serves a fresh app (fresh lifespan, fresh boot
    merge) without the process ever leaving the container — nobody restarts
    Aether by hand. A real SIGINT/SIGTERM never arms a restart, so the loop
    exits."""
    import os

    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    while True:
        server = uvicorn.Server(
            uvicorn.Config(create_app(), host="0.0.0.0", port=port, log_level="info")
        )
        restart.register(server)
        server.run()
        if not restart.take_pending():
            break
