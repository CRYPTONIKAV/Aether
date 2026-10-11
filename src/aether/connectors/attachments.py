"""User-shared files become memory: images described, documents read.

Perception-only by construction — this pipeline never registers an action
tool, and any action the agent later proposes from what it saw or read goes
through the same authz gate as everything else.

The connectors download the bytes (they own the platform credentials); this
module only thinks. The loop's drain calls it past the allowlist, so a
filtered sender's files are never seen.
"""

from __future__ import annotations

import base64
import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from ..llm.json_utils import extract_json
from ..llm.registry import ProviderRegistry
from ..llm.types import ImageBlock, Message
from ..memory.entities import Sender
from ..memory.events import EventStore, IngestResult
from .base import Attachment, InboundMessage

log = logging.getLogger("aether.connectors.attachments")

# connectors check the platform-reported size before downloading — one cap
# for every surface, above what the bot APIs hand over
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
# stored chars of extracted document text — the payload is read by salience
# once and clipped per turn, so this costs tokens exactly once per document
DOC_TEXT_CAP = 20_000

IMAGE_SYSTEM_PROMPT = (
    "You look at one image the user shared in chat — a photo, screenshot, or "
    "picture — and report what is in it. Perception only — never propose or "
    "take actions.\n"
    'Reply with strict JSON only: {"description": <one or two sentences about '
    "what the image shows>, "
    '"people": [<names or handles visible>], '
    '"text_visible": [<any text readable in the image>], '
    '"notable": [<anything unusual or worth remembering>]}'
)

_IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".heic": "image/heic",
}


@dataclass(frozen=True)
class AttachmentResult:
    """One processed attachment. `blind` marks an image Aether could not
    see (no vision-capable provider) — the drain turns that into the plain
    note."""

    ingested: IngestResult | None
    blind: bool = False


def attachments_note(results: list[AttachmentResult]) -> str:
    """The plain, in-chat words for images Aether couldn't see — "" when
    there is nothing to say. One note for a whole burst."""
    blind = sum(1 for r in results if r.blind)
    if not blind:
        return ""
    noun = "this image" if blind == 1 else "these images"
    return (
        f"I can't see {noun} yet — no vision model is set. Set one with "
        "/config (llm.vision_model), or pick a main model that can read "
        "images, and I'll see what you send."
    )


class AttachmentProcessor:
    """User-shared files in (connectors download the bytes), encrypted memory
    events out. Images are described by the vision provider — perception
    only; documents keep their extracted text so a later turn can answer
    questions about the content."""

    def __init__(self, providers: ProviderRegistry, events: EventStore) -> None:
        self._providers = providers
        self._events = events

    async def process(self, message: InboundMessage) -> list[AttachmentResult]:
        out: list[AttachmentResult] = []
        for attachment in message.attachments:
            if attachment.kind == "image":
                out.append(await self._image(attachment, message))
            else:
                out.append(await self._doc(attachment, message))
        return out

    async def _image(self, attachment: Attachment, message: InboundMessage) -> AttachmentResult:
        provider = self._providers.for_role("vision")
        if provider is None or not provider.supports_vision:
            log.warning("shared image dropped: no vision-capable provider configured")
            return AttachmentResult(ingested=None, blind=True)
        turn = await provider.complete(
            IMAGE_SYSTEM_PROMPT,
            [
                Message.user(
                    "What is in this image?"
                    + (f"\nThe user said: {message.text}" if message.text else ""),
                    images=[
                        ImageBlock(
                            data_b64=base64.b64encode(attachment.data).decode("ascii"),
                            media_type=_image_media_type(attachment),
                        )
                    ],
                )
            ],
            tools=[],
        )
        try:
            described = extract_json(turn.text)
            if not isinstance(described, dict):
                raise ValueError("vision turn returned a non-object")
        except Exception:
            # a vision turn that didn't produce JSON still carries
            # information — keep the raw text rather than lose the image
            log.warning("image vision returned non-JSON; storing raw text")
            described = {"description": (turn.text or "")[:1000]}
        payload = {k: v for k, v in described.items() if v not in (None, "", [])}
        payload["filename"] = attachment.filename
        payload["media_type"] = attachment.media_type
        if message.text:
            payload["caption"] = message.text
        payload["chat_ref"] = message.chat_ref
        ingested = await self._events.ingest(
            source=message.surface,
            kind="shared_image",
            payload=payload,
            sender=Sender(platform=message.surface, handle=message.handle),
        )
        return AttachmentResult(ingested=ingested)

    async def _doc(self, attachment: Attachment, message: InboundMessage) -> AttachmentResult:
        text = extract_doc_text(attachment.filename, attachment.media_type, attachment.data)
        payload: dict[str, str] = {
            "filename": attachment.filename,
            "media_type": attachment.media_type,
            "chat_ref": message.chat_ref,
        }
        if text:
            payload["text"] = text[:DOC_TEXT_CAP]
        else:
            # even an unreadable file leaves a row — the agent can honestly
            # recall that the file existed and what it was called
            payload["extract_note"] = "no readable text could be extracted"
        if message.text:
            payload["caption"] = message.text
        ingested = await self._events.ingest(
            source=message.surface,
            kind="shared_doc",
            payload=payload,
            sender=Sender(platform=message.surface, handle=message.handle),
        )
        return AttachmentResult(ingested=ingested)


def _image_media_type(attachment: Attachment) -> str:
    """A real image MIME for the provider adapters — from the platform,
    the filename, or a png default."""
    if (attachment.media_type or "").startswith("image/"):
        return attachment.media_type
    return _IMAGE_MEDIA_TYPES.get(Path(attachment.filename or "").suffix.lower(), "image/png")


# -- document text extraction --------------------------------------------------
# Best-effort plain text from a shared document; the office formats are zips
# of XML, so stdlib carries them. A corrupt or unknown file extracts to "" —
# extraction never raises.


def extract_doc_text(filename: str, media_type: str, data: bytes) -> str:
    try:
        ext = Path(filename or "").suffix.lower().lstrip(".")
        if ext == "pdf" or "pdf" in (media_type or ""):
            return _pdf_text(data)
        if ext == "docx":
            return _zip_text(data, "word/document.xml", _docx_text)
        if ext == "xlsx":
            return _xlsx_text(data)
        if ext == "pptx":
            return _pptx_text(data)
        if ext in ("txt", "csv", "md") or (media_type or "").startswith("text/"):
            return data.decode("utf-8", errors="replace")
        return ""
    except Exception:
        log.warning("text extraction failed for %s — continuing", filename, exc_info=True)
        return ""


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _zip_text(data: bytes, member: str, to_text) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as zf, zf.open(member) as f:
        return to_text(f.read())


_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DRAWING_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _docx_text(xml: bytes) -> str:
    root = ElementTree.fromstring(xml)
    lines = []
    for para in root.iter(f"{_WORD_NS}p"):
        line = "".join(t.text or "" for t in para.iter(f"{_WORD_NS}t")).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _xlsx_text(data: bytes) -> str:
    """Shared strings plus the literal cell values — a memory aid for
    "what did that sheet say", not a spreadsheet engine."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        parts: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.iter(f"{_SHEET_NS}si"):
                text = "".join(t.text or "" for t in si.iter(f"{_SHEET_NS}t")).strip()
                if text:
                    parts.append(text)
        for name in sorted(zf.namelist()):
            if name.startswith("xl/worksheets/") and name.endswith(".xml"):
                root = ElementTree.fromstring(zf.read(name))
                for value in root.iter(f"{_SHEET_NS}v"):
                    if value.text and value.text.strip():
                        parts.append(value.text.strip())
        return "\n".join(parts)


def _pptx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        slides = sorted(
            name
            for name in zf.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        lines = []
        for name in slides:
            root = ElementTree.fromstring(zf.read(name))
            text = " ".join((t.text or "").strip() for t in root.iter(f"{_DRAWING_NS}t")).strip()
            if text:
                lines.append(text)
        return "\n".join(lines)
