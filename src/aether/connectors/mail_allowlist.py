"""Mail read filtering: the Gmail allowlist, enforced on the results
themselves.

ingest() drops non-allowlisted senders before storage, but a bulk read
(`composio__GMAIL_FETCH_EMAILS` — the model's own calls and the 5-minute
poll) answers with a whole inbox at once, senderless from ingest's point
of view. This filter sits on the tool-result paths instead: it walks the
returned message batch and keeps only what the user's Gmail list and the
cross-app bucket allow — mail from an allowlisted sender, mail the user
sent to an allowlisted person, and self-addressed mail (from me to me),
which needs no knowledge of the account's own address to recognize.

Drops are silent to Aether — the model never sees the removed messages —
and the only trace is one audit row naming the count. Sender addresses
never reach the audit log.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..authz.audit import AuditLog
from ..config import ContactsConfig
from ..memory.entities import Sender
from ..memory.events import is_allowlisted

# 'Display Name <addr@example.com>' or a bare address — one shape reader
_ADDRESS_IN_ANGLE = re.compile(r"<([^<>@\s]+@[^<>\s]+)>")


def _addresses(value: Any) -> list[str]:
    """Pull the bare addresses out of a sender/to field, which arrives as
    'Name <addr>', a bare address, a comma-separated run of either — and
    any of those can be a list."""
    parts = value if isinstance(value, list) else [value]
    out: list[str] = []
    for part in parts:
        if not isinstance(part, str):
            continue
        for piece in part.split(","):
            found = _ADDRESS_IN_ANGLE.search(piece)
            addr = (found.group(1) if found else piece).strip().lower()
            if "@" in addr and addr not in out:
                out.append(addr)
    return out


class MailAllowlist:
    """Filters gmail tool results down to the allowlisted people."""

    def __init__(self, contacts: ContactsConfig, audit: AuditLog) -> None:
        self._contacts = contacts
        self._audit = audit

    async def filter_result(self, name: str, result: str) -> str:
        """One choke point for every execution path. Non-gmail calls and
        results we can't read pass through untouched."""
        if not name.startswith("composio__GMAIL_"):
            return result
        try:
            envelope = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
        messages = envelope.get("data", {}).get("messages") if isinstance(envelope, dict) else None
        if not isinstance(messages, list) or not messages:
            return result

        kept = [m for m in messages if self._keep(m)]
        dropped = len(messages) - len(kept)
        if dropped:
            await self._audit.append(
                actor="system",
                tool_name=name,
                decision="filtered",
                params={"messages_kept": len(kept), "messages_dropped": dropped},
                outcome=(
                    f"dropped {dropped} message{'' if dropped == 1 else 's'}"
                    " not on the Gmail allowlist"
                ),
            )
            envelope["data"]["messages"] = kept
            return json.dumps(envelope, ensure_ascii=False)
        return result

    def _keep(self, message: Any) -> bool:
        """The user's mail rule: keep self-addressed mail, mail from an
        allowlisted sender, and mail the user sent to an allowlisted
        person; everything else goes."""
        if not isinstance(message, dict):
            return True  # an unknown shape is never silently destroyed
        sender_addrs = _addresses(message.get("sender"))
        to_addrs = _addresses(message.get("to"))
        if not sender_addrs and not to_addrs:
            return True  # nothing to judge by — a shape we don't model yet
        if any(s in to_addrs for s in sender_addrs):
            return True  # from me to me
        # from an allowlisted sender, or mail the user sent an allowlisted person
        return any(
            is_allowlisted(self._contacts, "gmail", Sender(platform="email", handle=a))
            for a in sender_addrs + to_addrs
        )
