"""Shared-file tests — the attachment processor and document extraction.
FakeProvider + FakeEventStore; no network, no DB. (Pattern:
test_screenvision.py.)"""

from __future__ import annotations

import io
import zipfile

from fakes import FakeEventStore, FakeProvider, FakeRegistry

from aether.connectors.attachments import (
    AttachmentProcessor,
    AttachmentResult,
    attachments_note,
    extract_doc_parts,
    extract_doc_text,
)
from aether.connectors.base import Attachment, InboundMessage, classify_attachment
from aether.llm.types import ImageBlock, Turn

MINIMAL_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 24 Tf 100 700 Td (hello world) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
trailer
<< /Root 1 0 R >>
startxref
0
%%EOF
"""


def _message(*attachments: Attachment, text: str = "") -> InboundMessage:
    return InboundMessage(
        surface="telegram",
        handle="@vedant",
        text=text,
        chat_ref="42",
        attachments=tuple(attachments),
    )


def _image() -> Attachment:
    return Attachment("image", "cat.png", "image/png", b"\x89PNG-fake")


def _zip(parts: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in parts.items():
            zf.writestr(name, content)
    return buf.getvalue()


class TestClassify:
    def test_mime_says_image_first(self) -> None:
        assert classify_attachment("no_extension", "image/jpeg") == "image"

    def test_filename_extension_decides_without_a_mime(self) -> None:
        assert classify_attachment("shot.webp", "") == "image"
        assert classify_attachment("report.pdf", "") == "doc"

    def test_everything_else_is_a_doc(self) -> None:
        assert classify_attachment("archive.zip", "application/zip") == "doc"


class TestImages:
    async def test_an_image_is_described_and_stored(self) -> None:
        provider = FakeProvider(
            [
                Turn(
                    text='{"description": "a ginger cat on a windowsill", '
                    '"people": [], "text_visible": ["MILK 2%"], "notable": []}'
                )
            ]
        )
        processor = AttachmentProcessor(FakeRegistry(provider), FakeEventStore())
        results = await processor.process(_message(_image(), text="is this our cat?"))

        assert results[0].blind is False
        assert results[0].ingested is not None and results[0].ingested.stored is True
        record = processor._events.ingested[0]
        assert record["kind"] == "shared_image"
        assert record["source"] == "telegram"
        payload = record["payload"]
        assert payload["description"] == "a ginger cat on a windowsill"
        assert payload["text_visible"] == ["MILK 2%"]
        assert payload["caption"] == "is this our cat?"
        assert payload["filename"] == "cat.png"
        assert "people" not in payload  # empty values dropped
        sender = record["sender"]
        assert sender.platform == "telegram" and sender.handle == "@vedant"

        # the vision turn carried the actual image, and no tools
        system, messages, tools = provider.calls[0]
        assert tools == []
        assert "is this our cat?" in messages[0].text
        assert any(
            isinstance(b, ImageBlock) and b.media_type == "image/png" for b in messages[0].blocks
        )

    async def test_no_vision_provider_is_blind_never_crashes(self) -> None:
        processor = AttachmentProcessor(FakeRegistry(None), FakeEventStore())
        results = await processor.process(_message(_image(), text="look"))
        assert results == [AttachmentResult(ingested=None, blind=True)]
        assert processor._events.ingested == []

    async def test_a_blind_provider_is_blind_too(self) -> None:
        provider = FakeProvider([Turn(text="{}")])
        provider.supports_vision = False
        processor = AttachmentProcessor(FakeRegistry(provider), FakeEventStore())
        results = await processor.process(_message(_image()))
        assert results[0].blind is True
        assert provider.calls == []

    async def test_non_json_vision_turn_stores_raw_text(self) -> None:
        processor = AttachmentProcessor(
            FakeRegistry(FakeProvider([Turn(text="A handwritten note, no JSON")])),
            FakeEventStore(),
        )
        results = await processor.process(_message(_image()))
        assert results[0].blind is False
        assert processor._events.ingested[0]["payload"]["description"] == (
            "A handwritten note, no JSON"
        )

    def test_the_note_is_the_plain_words_once_per_burst(self) -> None:
        one = attachments_note([AttachmentResult(None, blind=True)])
        assert one == (
            "I can't see this image yet — no vision model is set. Set one with "
            "/config (llm.vision_model), or pick a main model that can read "
            "images, and I'll see what you send."
        )
        two = attachments_note(
            [AttachmentResult(None, blind=True), AttachmentResult(None, blind=True)]
        )
        assert "these images" in two
        assert attachments_note([AttachmentResult(None), AttachmentResult(None)]) == ""


class TestDocs:
    async def test_a_text_file_is_stored_with_its_content(self) -> None:
        processor = AttachmentProcessor(FakeRegistry(None), FakeEventStore())
        attachment = Attachment("doc", "notes.txt", "text/plain", b"remember the milk")
        results = await processor.process(_message(attachment))

        assert results[0].ingested is not None and results[0].ingested.stored is True
        record = processor._events.ingested[0]
        assert record["kind"] == "shared_doc"
        payload = record["payload"]
        assert payload["text"] == "remember the milk"
        assert payload["filename"] == "notes.txt"
        assert payload["chat_ref"] == "42"

    async def test_doc_text_is_capped(self) -> None:
        processor = AttachmentProcessor(FakeRegistry(None), FakeEventStore())
        big = Attachment("doc", "big.txt", "text/plain", b"x" * 100_000)
        await processor.process(_message(big))
        assert len(processor._events.ingested[0]["payload"]["text"]) == 20_000

    async def test_an_unextractable_file_still_leaves_a_row(self) -> None:
        processor = AttachmentProcessor(FakeRegistry(None), FakeEventStore())
        attachment = Attachment("doc", "archive.zip", "application/zip", b"not really a zip")
        await processor.process(_message(attachment))
        payload = processor._events.ingested[0]["payload"]
        assert payload["extract_note"] == "no readable text could be extracted"
        assert "text" not in payload


class TestExtraction:
    def test_plain_text_and_family(self) -> None:
        assert extract_doc_text("a.txt", "text/plain", b"hello\nworld") == "hello\nworld"
        assert extract_doc_text("a.csv", "", b"a,b") == "a,b"
        assert extract_doc_text("a.md", "text/markdown", b"# hi") == "# hi"

    def test_pdf(self) -> None:
        text = extract_doc_text("letter.pdf", "application/pdf", MINIMAL_PDF)
        assert "hello world" in text

    def test_docx(self) -> None:
        xml = (
            b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            b"<w:body>"
            b"<w:p><w:r><w:t>First </w:t></w:r><w:r><w:t>paragraph</w:t></w:r></w:p>"
            b"<w:p><w:r><w:t>Second</w:t></w:r></w:p>"
            b"</w:body></w:document>"
        )
        docx = _zip({"word/document.xml": xml})
        assert extract_doc_text("letter.docx", "", docx) == "First paragraph\nSecond"

    def test_xlsx(self) -> None:
        strings = (
            b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            b"<si><t>Name</t></si><si><t>Sam</t></si></sst>"
        )
        sheet = (
            b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            b"<sheetData><row><c><v>42</v></c></row></sheetData></worksheet>"
        )
        xlsx = _zip({"xl/sharedStrings.xml": strings, "xl/worksheets/sheet1.xml": sheet})
        text = extract_doc_text("books.xlsx", "", xlsx)
        assert "Name" in text and "Sam" in text and "42" in text

    def test_pptx(self) -> None:
        slide1 = (
            b'<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            b'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            b"<p:cSld><p:spTree><p:sp><p:txBody>"
            b"<a:p><a:r><a:t>Roadmap</a:t></a:r></a:p>"
            b"</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
        )
        slide2 = (
            b'<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            b'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            b"<p:cSld><p:spTree><p:sp><p:txBody>"
            b"<a:p><a:r><a:t>Q3 plan</a:t></a:r></a:p>"
            b"</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
        )
        pptx = _zip({"ppt/slides/slide2.xml": slide2, "ppt/slides/slide1.xml": slide1})
        assert extract_doc_text("deck.pptx", "", pptx) == "Roadmap\nQ3 plan"

    def test_corrupt_files_extract_to_nothing_never_raise(self) -> None:
        garbage = b"\x00\x01garbage\xff\xfe"
        for name in ("a.pdf", "a.docx", "a.xlsx", "a.pptx", "a.zip", ""):
            assert extract_doc_text(name, "", garbage) == ""
        assert extract_doc_text("a.txt", "", garbage) != ""  # plain text survives anything


class TestStructuredParts:
    """extract_doc_parts: the (location, text) form the vector memory
    indexes — pages and slides carry the numbering the file itself has,
    everything else is one whole part."""

    def test_pdf_parts_are_per_page(self) -> None:
        parts = extract_doc_parts("letter.pdf", "application/pdf", MINIMAL_PDF)
        assert parts, "the minimal pdf has at least one page"
        assert parts[0][0] == "page 1"
        assert "hello world" in parts[0][1]

    def test_pptx_parts_are_per_slide_in_deck_order(self) -> None:
        slide = (
            b'<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            b'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            b"<p:cSld><p:spTree><p:sp><p:txBody>"
            b"<a:p><a:r><a:t>Roadmap</a:t></a:r></a:p>"
            b"</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
        )
        # slide10 would sort before slide2 lexicographically — numeric
        # order is what the deck actually shows
        pptx = _zip(
            {
                "ppt/slides/slide10.xml": slide,
                "ppt/slides/slide2.xml": slide,
                "ppt/slides/slide1.xml": slide,
            }
        )
        parts = extract_doc_parts("deck.pptx", "", pptx)
        assert [loc for loc, _ in parts] == ["slide 1", "slide 2", "slide 10"]
        assert all("Roadmap" in text for _, text in parts)

    def test_plain_text_is_one_whole_part(self) -> None:
        assert extract_doc_parts("a.txt", "text/plain", b"hello\nworld") == [("", "hello\nworld")]
        assert extract_doc_parts("a.csv", "", b"a,b") == [("", "a,b")]

    def test_unreadable_file_has_no_parts(self) -> None:
        garbage = b"\x00\x01garbage\xff\xfe"
        assert extract_doc_parts("a.pdf", "", garbage) == []
        assert extract_doc_parts("a.docx", "", garbage) == []
