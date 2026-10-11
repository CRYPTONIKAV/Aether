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

    def test_pdf_missing_dependency_is_distinguishable_and_actionable(
        self, monkeypatch, caplog
    ) -> None:
        """A missing pypdf must not look like a corrupt file: it's logged
        loudly (ERROR, naming the fix) instead of blending into the quiet
        per-file WARNING every other extraction failure gets."""
        import sys

        monkeypatch.setitem(sys.modules, "pypdf", None)  # simulates "not installed"
        with caplog.at_level("ERROR", logger="aether.connectors.attachments"):
            text = extract_doc_text("letter.pdf", "application/pdf", MINIMAL_PDF)
        assert text == ""
        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert error_records, "missing pypdf must log at ERROR, not just the usual WARNING"
        assert "pypdf" in error_records[0].message
        assert "install" in error_records[0].message.lower()

    def test_pdf_unexpected_parsing_errors_still_degrade_quietly(self, caplog) -> None:
        """A genuine parsing failure (not a missing dependency) must keep
        its existing, quieter treatment — only the dependency case gets the
        louder log level."""
        with caplog.at_level("WARNING", logger="aether.connectors.attachments"):
            text = extract_doc_text("letter.pdf", "application/pdf", b"not a pdf at all")
        assert text == ""
        assert not any(r.levelname == "ERROR" for r in caplog.records)
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_pdf_encrypted_or_unreadable_extracts_to_nothing_never_raises(self) -> None:
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(io.BytesIO(MINIMAL_PDF))
        writer = PdfWriter()
        writer.append(reader)
        writer.encrypt(user_password="secret", owner_password="secret-owner")
        buf = io.BytesIO()
        writer.write(buf)

        assert extract_doc_text("locked.pdf", "application/pdf", buf.getvalue()) == ""

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


class TestZipBombProtection:
    """docx/xlsx/pptx are zips of XML — MAX_ATTACHMENT_BYTES only bounds the
    compressed upload, so these guard the decompressed side: a small
    crafted archive must never be allowed to blow up memory/CPU."""

    def _bomb_zip(self, member: str, size: int) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(member, b"A" * size)  # deflates to a tiny compressed size
        return buf.getvalue()

    def test_oversized_member_is_rejected_before_full_decompression(self) -> None:
        from aether.connectors.attachments import MAX_ZIP_ENTRY_BYTES

        bomb = self._bomb_zip("word/document.xml", MAX_ZIP_ENTRY_BYTES + 1024)
        assert extract_doc_text("letter.docx", "", bomb) == ""

    def test_many_small_members_are_capped_by_the_total_budget(self) -> None:
        from aether.connectors.attachments import MAX_ZIP_ENTRY_BYTES, MAX_ZIP_TOTAL_BYTES

        # each member is under the per-entry cap, but enough of them blow
        # past the combined-document cap
        per_member = MAX_ZIP_ENTRY_BYTES // 2
        count = (MAX_ZIP_TOTAL_BYTES // per_member) + 2
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("xl/sharedStrings.xml", b"<sst/>")
            for i in range(count):
                zf.writestr(f"xl/worksheets/sheet{i}.xml", b"A" * per_member)
        assert extract_doc_text("book.xlsx", "", buf.getvalue()) == ""

    def test_too_many_archive_entries_is_rejected(self) -> None:
        from aether.connectors.attachments import MAX_ZIP_ENTRIES

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for i in range(MAX_ZIP_ENTRIES + 1):
                zf.writestr(f"xl/worksheets/sheet{i}.xml", b"<x/>")
        assert extract_doc_text("book.xlsx", "", buf.getvalue()) == ""

    def test_xml_doctype_and_entity_declarations_are_rejected(self) -> None:
        """The 'billion laughs' shape: a tiny entity declaration that
        expands to an enormous parse — refused outright, not expanded."""
        xml = (
            b'<?xml version="1.0"?>'
            b'<!DOCTYPE lolz [<!ENTITY lol "lol">]>'
            b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            b"<w:body><w:p><w:r><w:t>&lol;</w:t></w:r></w:p></w:body></w:document>"
        )
        docx = _zip({"word/document.xml": xml})
        assert extract_doc_text("letter.docx", "", docx) == ""

    def test_normal_documents_are_unaffected_by_the_new_caps(self) -> None:
        """The bounded reader must still produce identical output for an
        ordinary, well-formed document — these are the same fixtures
        TestExtraction uses for docx/xlsx/pptx."""
        xml = (
            b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            b"<w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p></w:body></w:document>"
        )
        docx = _zip({"word/document.xml": xml})
        assert extract_doc_text("letter.docx", "", docx) == "Hello"
