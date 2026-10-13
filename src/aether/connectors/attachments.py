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
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from ..llm.json_utils import extract_json
from ..llm.registry import ProviderRegistry
from ..llm.types import ImageBlock, Message
from ..memory.entities import Sender
from ..memory.events import EventStore, IngestResult
from ..memory.vectors import VectorMemory
from .base import Attachment, InboundMessage

log = logging.getLogger("aether.connectors.attachments")

# connectors check the platform-reported size before downloading — one cap
# for every surface, above what the bot APIs hand over
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
# stored chars of extracted document text — the payload is read by salience
# once and clipped per turn, so this costs tokens exactly once per document
DOC_TEXT_CAP = 20_000


def reported_size_exceeds_cap(size: int | None, cap: int) -> bool:
    """True only when the platform told us, unambiguously, that the file is
    too big. A missing, zero, or negative size is never treated as proof of
    safety — some bot APIs (Telegram's documented Optional file_size among
    them) omit it, and a connector must not read "absent" as "small". Every
    connector still checks the actual downloaded bytes against the same cap
    once the file is in hand, regardless of what this says."""
    return isinstance(size, int) and size > cap


def actual_size_exceeds_cap(data: bytes, cap: int) -> bool:
    """The one check that can't be lied to by platform metadata — run after
    every download, whatever the reported size claimed."""
    return len(data) > cap


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

    def __init__(
        self,
        providers: ProviderRegistry,
        events: EventStore,
        indexer: VectorMemory | None = None,
    ) -> None:
        self._providers = providers
        self._events = events
        self._indexer = indexer

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
        # the event payload keeps only a preview — the vector chunks carry
        # the *full* text, page by page, slide by slide. A failure here
        # leaves indexed_at unset, so the loop's backfill retries later.
        if self._indexer is not None and ingested.stored and ingested.event_id is not None:
            try:
                await self._indexer.index_doc(
                    ingested.event_id,
                    source=message.surface,
                    filename=attachment.filename,
                    parts=extract_doc_parts(
                        attachment.filename, attachment.media_type, attachment.data
                    ),
                )
            except Exception:
                log.warning(
                    "document indexing failed for %s — backfill will retry",
                    attachment.filename,
                    exc_info=True,
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


class _DependencyMissing(RuntimeError):
    """A required parsing dependency isn't installed in this environment —
    distinguishes a broken/unsynced deployment from a genuinely unreadable
    file, which get very different log treatment below."""


class _ZipBombSuspected(RuntimeError):
    """A zip-based document (docx/xlsx/pptx) decompressed past one of the
    hard ceilings below, or declared more entries/entity definitions than a
    real office file ever needs. Treated like any other unreadable file —
    logged and degraded to "", never allowed to run away with memory/CPU."""


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
    except _ZipBombSuspected as exc:
        log.warning("text extraction for %s refused — %s", filename, exc)
        return ""
    except _DependencyMissing as exc:
        # loud and actionable: this is an out-of-sync environment, not a bad
        # file — every PDF would otherwise silently "extract" to nothing
        # with no clue why, indistinguishable from a corrupt upload
        log.error(
            "text extraction for %s needs a dependency that isn't installed "
            "(%s) — the environment is out of sync with pyproject.toml",
            filename,
            exc,
        )
        return ""
    except Exception:
        log.warning("text extraction failed for %s — continuing", filename, exc_info=True)
        return ""


def extract_doc_parts(filename: str, media_type: str, data: bytes) -> list[tuple[str, str]]:
    """The same text as extract_doc_text, but as `(location, text)` parts a
    document keeps its structure in: one part per pdf page, one per slide.
    This is what makes "what's in slide 3" a question with an answer — and
    what the vector memory indexes, so every chunk knows where it lived.
    Unstructured formats are one whole part; size-splitting happens later on
    paragraph boundaries."""
    try:
        ext = Path(filename or "").suffix.lower().lstrip(".")
        if ext == "pdf" or "pdf" in (media_type or ""):
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            return [
                (f"page {i}", page.extract_text() or "") for i, page in enumerate(reader.pages, 1)
            ]
        if ext == "pptx":
            return _pptx_parts(data)
        # everything else is one whole part: retrieval is semantic, so
        # plain paragraph-boundary chunking finds the right passage — and
        # only pdf/pptx carry a real location worth citing
        text = extract_doc_text(filename, media_type, data)
        return [("", text)] if text else []
    except Exception:
        log.warning("structured extraction failed for %s — continuing", filename, exc_info=True)
        return []


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise _DependencyMissing(f"pypdf is not installed ({exc}); run: pip install pypdf") from exc

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# Hard ceilings for zip-based office documents. A decompression bomb hides a
# huge payload behind a tiny compressed size — the zip's own central
# directory is attacker-controlled and never trusted as the sole guard, so
# every member is read in chunks with a running check against these caps
# rather than pulled fully into memory first and measured afterward.
MAX_ZIP_ENTRIES = 2_000
MAX_ZIP_ENTRY_BYTES = 10 * 1024 * 1024  # one decompressed member
MAX_ZIP_TOTAL_BYTES = 30 * 1024 * 1024  # every member read for one document
_ZIP_READ_CHUNK = 64 * 1024


def _open_bounded_zip(data: bytes) -> zipfile.ZipFile:
    zf = zipfile.ZipFile(io.BytesIO(data))
    if len(zf.infolist()) > MAX_ZIP_ENTRIES:
        raise _ZipBombSuspected(f"{len(zf.infolist())} archive entries exceeds the cap")
    return zf


def _read_bounded(zf: zipfile.ZipFile, name: str, total_read: list[int]) -> bytes:
    """One member's bytes, decompressed in chunks so an oversized member is
    caught well before a single `.read()` call would buffer the whole
    thing. `total_read` is a one-item list shared across every member read
    for the same document, so the combined decompressed size of a
    many-small-members bomb is capped too, not just any single member."""
    info = zf.getinfo(name)
    # the declared size is a hint, not a guarantee (it can under-report
    # what actually decompresses) — it only lets an honest giant fail fast
    if info.file_size > MAX_ZIP_ENTRY_BYTES:
        raise _ZipBombSuspected(f"{name} declares {info.file_size} bytes, over the per-entry cap")
    out = bytearray()
    with zf.open(name) as member:
        while chunk := member.read(_ZIP_READ_CHUNK):
            out += chunk
            total_read[0] += len(chunk)
            if len(out) > MAX_ZIP_ENTRY_BYTES:
                raise _ZipBombSuspected(f"{name} decompressed past the per-entry cap")
            if total_read[0] > MAX_ZIP_TOTAL_BYTES:
                raise _ZipBombSuspected("document decompressed past the total cap")
    return bytes(out)


def _reject_doctype(xml: bytes) -> None:
    """Office XML parts never legitimately declare a DOCTYPE or custom
    entities — stdlib expat expands entities by default, which turns a few
    bytes of declarations into an unbounded parse (the "billion laughs"
    shape). A DOCTYPE must precede the root element in well-formed XML, so
    a check of the first few KB is enough to catch one; this is used
    instead of reaching into ElementTree's internals, which no longer
    expose expat's own handlers in current Python (the C accelerator
    doesn't carry them)."""
    if b"<!DOCTYPE" in xml[:4096] or b"<!ENTITY" in xml[:4096]:
        raise _ZipBombSuspected("XML DOCTYPE/entity declarations are not allowed")


def _parse_xml(xml: bytes) -> ElementTree.Element:
    if len(xml) > MAX_ZIP_ENTRY_BYTES:  # belt-and-braces; callers already cap this
        raise _ZipBombSuspected("XML member too large to parse")
    _reject_doctype(xml)
    return ElementTree.fromstring(xml)


def _zip_text(data: bytes, member: str, to_text) -> str:
    with _open_bounded_zip(data) as zf:
        return to_text(_read_bounded(zf, member, [0]))


_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DRAWING_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _docx_text(xml: bytes) -> str:
    root = _parse_xml(xml)
    lines = []
    for para in root.iter(f"{_WORD_NS}p"):
        line = "".join(t.text or "" for t in para.iter(f"{_WORD_NS}t")).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _xlsx_text(data: bytes) -> str:
    """Shared strings plus the literal cell values — a memory aid for
    "what did that sheet say", not a spreadsheet engine."""
    with _open_bounded_zip(data) as zf:
        total_read = [0]
        parts: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = _parse_xml(_read_bounded(zf, "xl/sharedStrings.xml", total_read))
            for si in root.iter(f"{_SHEET_NS}si"):
                text = "".join(t.text or "" for t in si.iter(f"{_SHEET_NS}t")).strip()
                if text:
                    parts.append(text)
        for name in sorted(zf.namelist()):
            if name.startswith("xl/worksheets/") and name.endswith(".xml"):
                root = _parse_xml(_read_bounded(zf, name, total_read))
                for value in root.iter(f"{_SHEET_NS}v"):
                    if value.text and value.text.strip():
                        parts.append(value.text.strip())
        return "\n".join(parts)


def _pptx_text(data: bytes) -> str:
    with _open_bounded_zip(data) as zf:
        total_read = [0]
        slides = sorted(
            name
            for name in zf.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        lines = []
        for name in slides:
            root = _parse_xml(_read_bounded(zf, name, total_read))
            text = " ".join((t.text or "").strip() for t in root.iter(f"{_DRAWING_NS}t")).strip()
            if text:
                lines.append(text)
        return "\n".join(lines)


def _pptx_parts(data: bytes) -> list[tuple[str, str]]:
    """One `(slide N, text)` part per slide, in true deck order — slide10
    must not sort before slide2 the way a filename sort would."""
    slide_re = re.compile(r"ppt/slides/slide(\d+)\.xml$")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        slides = sorted(
            (name for name in zf.namelist() if slide_re.search(name)),
            key=lambda name: int(slide_re.search(name).group(1)),  # type: ignore[union-attr]
        )
        parts: list[tuple[str, str]] = []
        for name in slides:
            root = ElementTree.fromstring(zf.read(name))
            text = " ".join((t.text or "").strip() for t in root.iter(f"{_DRAWING_NS}t")).strip()
            if text:
                parts.append((f"slide {slide_re.search(name).group(1)}", text))  # type: ignore[union-attr]
        return parts
