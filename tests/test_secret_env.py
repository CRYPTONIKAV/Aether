"""The secrets-by-name resolver — names in, values out, nothing recorded.

The /apps promise rests on this module: chat carries a key's name and
never its value, config references keys as $NAME, and every resolve is
fresh (the .env file re-read on demand), so a key added after boot needs
no restart. Precedence is the whole design: the paste store (a paste is
the most recent deliberate act), then the environment, then the file.
These tests pin the expansion syntax, that precedence, and the
missing-name contract. Hermetic by construction: every resolver points
at a tmp .env and a fake paste store.
"""

from __future__ import annotations

import re

import pytest

from aether.secret_env import EnvResolver, SecretEnvError


class FakeSecretStore:
    """The SecretStore double — the pastes the resolver would find."""

    def __init__(self, secrets: dict[str, str] | None = None) -> None:
        self.secrets = dict(secrets or {})

    async def get(self, name: str) -> str | None:
        return self.secrets.get(name)

    async def names(self) -> set[str]:
        return set(self.secrets)


def _resolver(
    tmp_path,
    env_text: str = "",
    secrets: dict[str, str] | None = None,
) -> EnvResolver:
    env_file = tmp_path / ".env"
    env_file.write_text(env_text)
    return EnvResolver(env_file, secrets=FakeSecretStore(secrets))


# -- expansion syntax --------------------------------------------------------------


async def test_both_reference_spellings_expand(tmp_path) -> None:
    resolver = _resolver(tmp_path, env_text="KEY=secret\n")
    assert await resolver.expand_value("Bearer $KEY") == "Bearer secret"
    assert await resolver.expand_value("Bearer ${KEY}") == "Bearer secret"
    assert await resolver.expand_value("$KEY and $KEY") == "secret and secret"
    assert await resolver.expand_value("") == ""


async def test_a_lone_dollar_and_lowercase_refs_stay_literal(tmp_path) -> None:
    """Prices and shell-speak are not references: a bare $, $5, and a
    lowercase $name all pass through untouched."""
    resolver = _resolver(tmp_path, env_text="KEY=secret\n")
    assert await resolver.expand_value("costs $5") == "costs $5"
    assert await resolver.expand_value("$home") == "$home"
    assert await resolver.expand_value("no refs here") == "no refs here"


async def test_expand_map_resolves_every_value(tmp_path) -> None:
    resolver = _resolver(tmp_path, env_text="A=1\nB=2\n")
    resolved = await resolver.expand_map(
        {"TOKEN": "$A", "Authorization": "Bearer ${B}", "PLAIN": "as-is"}
    )
    assert resolved == {"TOKEN": "1", "Authorization": "Bearer 2", "PLAIN": "as-is"}


# -- precedence and the missing-name contract ---------------------------------------


async def test_the_real_environment_wins_over_the_env_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PRECEDENCE_KEY", "from-the-environment")
    resolver = _resolver(
        tmp_path, env_text="PRECEDENCE_KEY=from-the-file\nFILE_ONLY=from-the-file\n"
    )
    assert await resolver.resolve("PRECEDENCE_KEY") == "from-the-environment"
    assert await resolver.resolve("FILE_ONLY") == "from-the-file"


async def test_a_missing_name_resolves_to_none(tmp_path) -> None:
    resolver = _resolver(tmp_path)  # no .env contents, no paste-store rows
    assert await resolver.resolve("NOT_SET_ANYWHERE") is None


async def test_expanding_a_missing_name_raises_with_the_name_in_plain_words(tmp_path) -> None:
    resolver = _resolver(tmp_path)
    with pytest.raises(SecretEnvError) as exc:
        await resolver.expand_value("Bearer $NOT_SET_ANYWHERE")
    assert "NOT_SET_ANYWHERE" in str(exc.value)
    assert "isn't set" in str(exc.value)
    assert "the moment" in str(exc.value)  # the retry promise, in the same breath


# -- the paste store's precedence ----------------------------------------------------


async def test_a_pasted_key_wins_over_the_environ_and_the_file(monkeypatch, tmp_path) -> None:
    """A paste is the most recent deliberate act: the store outranks the
    real environment and the older .env line — the re-paste of a rotated
    token works without touching a file."""
    monkeypatch.setenv("PASTED_KEY", "from-the-environment")
    resolver = _resolver(
        tmp_path,
        env_text="PASTED_KEY=from-the-file\n",
        secrets={"PASTED_KEY": "the-paste"},
    )
    assert await resolver.resolve("PASTED_KEY") == "the-paste"


async def test_the_store_is_consulted_only_for_env_var_shaped_names(tmp_path) -> None:
    """Nothing outside the $NAME grammar can ever be shadowed by a paste —
    a lowercase name reads the file like it always did, even if a store
    row spells the same word."""
    resolver = _resolver(tmp_path, env_text="lower=from-the-file\n", secrets={"lower": "the-paste"})
    assert await resolver.resolve("lower") == "from-the-file"


# -- presence checks: names only, never values ---------------------------------------


async def test_known_names_reports_presence_from_all_three_sources(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SOME_TEST_ENV_NAME", "a-value-nobody-should-echo")
    env_file = tmp_path / ".env"
    env_file.write_text("FROM_FILE=yes\nEMPTY=\n")
    resolver = EnvResolver(env_file, secrets=FakeSecretStore({"FROM_PASTE": "a-value"}))

    names = await resolver.known_names()
    assert "FROM_FILE" in names
    assert "SOME_TEST_ENV_NAME" in names
    assert "FROM_PASTE" in names
    assert "EMPTY" not in names  # an empty line is not a set key
    # env-var-looking names only — PATH & friends don't masquerade as app keys
    assert all(re.fullmatch(r"[A-Z][A-Z0-9_]*", name) for name in names)
    assert all(name not in names for name in ("", "lower_case"))


# -- the staleness fix ----------------------------------------------------------------


async def test_a_key_added_to_the_file_after_boot_resolves(tmp_path) -> None:
    """The house principle's proof: acts read the resolver, and the
    resolver re-reads .env — a key that lands later is simply there."""
    env_file = tmp_path / ".env"
    env_file.write_text("")
    resolver = EnvResolver(env_file)

    assert await resolver.resolve("LATE_KEY") is None
    env_file.write_text("LATE_KEY=now-its-here\n")
    assert await resolver.resolve("LATE_KEY") == "now-its-here"
    assert "LATE_KEY" in await resolver.known_names()
