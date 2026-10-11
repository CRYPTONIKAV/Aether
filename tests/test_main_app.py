"""App-level routes — the ones create_app pins on the app itself, not the
api_router: the pinger's plain 200s and the OpenAI-style capability answer."""

from __future__ import annotations

import httpx

from aether.main import create_app


def _client(tmp_path, llm: dict) -> httpx.AsyncClient:
    config = tmp_path / "config.yaml"
    config.write_text(f"llm: {llm}\n", encoding="utf-8")
    app = create_app(config_path=str(config))
    # no lifespan — these routes owe nothing to the stores it wires
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_health_is_a_plain_unguarded_200(tmp_path) -> None:
    async with _client(tmp_path, {"provider": "anthropic", "model": "claude-opus-5"}) as client:
        for path in ("/healthz", "/health"):
            response = await client.get(path)
            assert response.status_code == 200
            assert response.json() == {"ok": True}


async def test_models_answers_the_openai_compatible_probe(tmp_path) -> None:
    async with _client(
        tmp_path,
        {
            "provider": "ollama",
            "model": "llama3",
            "vision_model": "llava",
            "salience_model": "llama3",
        },
    ) as client:
        response = await client.get("/v1/models")
        assert response.status_code == 200
        body = response.json()
        # the OpenAI wire shape: a list of model objects, and every
        # configured model appears exactly once no matter how many
        # roles share it
        assert body["object"] == "list"
        assert [m["id"] for m in body["data"]] == ["llama3", "llava"]
        assert all(m["object"] == "model" for m in body["data"])
        assert all(m["owned_by"] == "ollama" for m in body["data"])


async def test_models_answers_with_the_default_config_too(tmp_path) -> None:
    async with _client(tmp_path, {"provider": "anthropic"}) as client:
        response = await client.get("/v1/models")
        assert response.status_code == 200
        assert [m["id"] for m in response.json()["data"]] == ["claude-opus-5"]
