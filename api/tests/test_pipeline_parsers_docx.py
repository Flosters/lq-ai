"""Unit tests for the DOCX parser (``parse_docx``, ADR 0017).

The parser shells out to Pandoc (``--track-changes=all``), so every test
that exercises extraction is skipped when the ``pandoc`` binary is not on
PATH — mirroring how ``test_pipeline_parsers.py`` skips without PyMuPDF.
The MIME/filename routing helpers are pure and always run.

The load-bearing assertions, per ADR 0017:

1. **Offset fidelity** — ``parse_docx → chunk_document`` and every chunk's
   ``[char_offset_start:char_offset_end]`` slice equals its ``content``.
2. **Redline policy** — ``canonical_text`` is the *changes-accepted* text
   (insertions kept, deletions dropped), byte-identical to a separate
   ``--track-changes=accept`` Pandoc pass; the revision layer retains every
   insertion/deletion/comment with author + date + a char anchor.

Fixtures are built in-test as minimal OOXML packages (``zipfile`` +
hand-written ``document.xml``) so the tracked-changes / comment markup is
explicit and reviewable — no binary fixture files.
"""

from __future__ import annotations

import io
import shutil
import zipfile

import pytest

from app.pipeline.chunker import chunk_document
from app.pipeline.parsers import (
    ParserDecodeError,
    ParserError,
    is_docx_filename,
    is_docx_mime,
    parse_docx,
)

requires_pandoc = pytest.mark.skipif(
    shutil.which("pandoc") is None,
    reason="pandoc binary not on PATH (ADR 0017 pins it in the api/ingest-worker images)",
)

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# ---------------------------------------------------------------------------
# OOXML fixture builder
# ---------------------------------------------------------------------------

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  {comments_override}
</Types>
"""

_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

_DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {comment_rel}
</Relationships>
"""

_DOCUMENT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
{body}
  </w:body>
</w:document>
"""

_COMMENTS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
{comments}
</w:comments>
"""


def _para(text: str) -> str:
    return f'    <w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'


def _docx_bytes(body_xml: str, comments_xml: str | None = None) -> bytes:
    """Assemble a minimal but valid .docx package from body XML."""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        comments_override = ""
        comment_rel = ""
        if comments_xml is not None:
            comments_override = (
                '<Override PartName="/word/comments.xml" ContentType='
                '"application/vnd.openxmlformats-officedocument.'
                'wordprocessingml.comments+xml"/>'
            )
            comment_rel = (
                '<Relationship Id="rId9" Type="http://schemas.openxmlformats.org/'
                'officeDocument/2006/relationships/comments" Target="comments.xml"/>'
            )
            zf.writestr("word/comments.xml", comments_xml)
        zf.writestr(
            "[Content_Types].xml",
            _CONTENT_TYPES.format(comments_override=comments_override),
        )
        zf.writestr("_rels/.rels", _ROOT_RELS)
        zf.writestr("word/_rels/document.xml.rels", _DOC_RELS.format(comment_rel=comment_rel))
        zf.writestr("word/document.xml", _DOCUMENT.format(body=body_xml))
    return buf.getvalue()


def _plain_docx() -> bytes:
    return _docx_bytes(
        "\n".join(
            [
                _para("Cláusula 1. Objeto del contrato."),
                _para("Cláusula 2. Precio: USD 100 mensuales."),
            ]
        )
    )


def _redlined_docx() -> bytes:
    """One paragraph with an insertion replacing a deletion (a live redline)."""

    body = """    <w:p>
      <w:r><w:t xml:space="preserve">El honorario mensual es </w:t></w:r>
      <w:ins w:id="1" w:author="Laura Fernández" w:date="2026-01-02T10:00:00Z">
        <w:r><w:t xml:space="preserve">USD 200</w:t></w:r>
      </w:ins>
      <w:del w:id="2" w:author="Laura Fernández" w:date="2026-01-02T10:00:00Z">
        <w:r><w:delText xml:space="preserve">USD 100</w:delText></w:r>
      </w:del>
      <w:r><w:t xml:space="preserve">.</w:t></w:r>
    </w:p>"""
    return _docx_bytes(body)


def _paragraph_redlined_docx() -> bytes:
    """A whole inserted paragraph and a whole deleted paragraph — the shape
    every real contract redline has. The deleted paragraph must vanish from
    the canonical stream *including* its block separator (the accept-pass
    equality test enforces byte-identity)."""

    body = """    <w:p><w:r><w:t xml:space="preserve">Antes.</w:t></w:r></w:p>
    <w:p>
      <w:pPr><w:rPr><w:ins w:id="9" w:author="Laura" w:date="2026-01-02T00:00:00Z"/></w:rPr></w:pPr>
      <w:ins w:id="10" w:author="Laura" w:date="2026-01-02T00:00:00Z">
        <w:r><w:t xml:space="preserve">Cláusula nueva entera.</w:t></w:r>
      </w:ins>
    </w:p>
    <w:p>
      <w:pPr><w:rPr><w:del w:id="11" w:author="Laura" w:date="2026-01-02T00:00:00Z"/></w:rPr></w:pPr>
      <w:del w:id="12" w:author="Laura" w:date="2026-01-02T00:00:00Z">
        <w:r><w:delText xml:space="preserve">Cláusula borrada entera.</w:delText></w:r>
      </w:del>
    </w:p>
    <w:p><w:r><w:t xml:space="preserve">Después.</w:t></w:r></w:p>"""
    return _docx_bytes(body)


def _commented_docx() -> bytes:
    """A paragraph whose middle run carries an anchored Word comment."""

    body = """    <w:p>
      <w:r><w:t xml:space="preserve">La parte podrá rescindir </w:t></w:r>
      <w:commentRangeStart w:id="0"/>
      <w:r><w:t xml:space="preserve">sin causa</w:t></w:r>
      <w:commentRangeEnd w:id="0"/>
      <w:r><w:commentReference w:id="0"/></w:r>
      <w:r><w:t xml:space="preserve"> con 30 días de aviso.</w:t></w:r>
    </w:p>"""
    comments = """  <w:comment w:id="0" w:author="Martín Pérez" w:date="2026-01-03T09:00:00Z">
    <w:p><w:r><w:t xml:space="preserve">Pedir causa justificada acá.</w:t></w:r></w:p>
  </w:comment>"""
    return _docx_bytes(body, comments_xml=_COMMENTS.format(comments=comments))


# ---------------------------------------------------------------------------
# Routing helpers (pure — no pandoc needed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_docx_mime_accepts_the_ooxml_mime() -> None:
    assert is_docx_mime(DOCX_MIME) is True
    assert is_docx_mime(DOCX_MIME.upper()) is True
    # Some uploaders append parameters; the bare type matches.
    assert is_docx_mime(f"{DOCX_MIME}; charset=binary") is True


@pytest.mark.unit
def test_is_docx_mime_rejects_non_docx() -> None:
    assert is_docx_mime("application/pdf") is False
    assert is_docx_mime("application/msword") is False  # legacy .doc — out of scope
    assert is_docx_mime("application/octet-stream") is False
    assert is_docx_mime("") is False


@pytest.mark.unit
def test_is_docx_filename_extension_fallback() -> None:
    assert is_docx_filename("contrato.docx") is True
    assert is_docx_filename("CONTRATO.DOCX") is True
    assert is_docx_filename("contrato.doc") is False  # legacy binary format
    assert is_docx_filename("contrato.pdf") is False
    assert is_docx_filename("") is False


# ---------------------------------------------------------------------------
# Extraction (pandoc-backed)
# ---------------------------------------------------------------------------


@requires_pandoc
@pytest.mark.unit
def test_plain_docx_paragraphs_survive_into_canonical_text() -> None:
    parsed = parse_docx(_plain_docx())

    assert "Cláusula 1. Objeto del contrato." in parsed.canonical_text
    assert "Cláusula 2. Precio: USD 100 mensuales." in parsed.canonical_text
    assert parsed.parser == "pandoc"
    assert parsed.parser_version.startswith("pandoc=")
    # DOCX has no fixed pagination — one synthetic page over the stream.
    assert parsed.page_count == 1
    assert parsed.pages[0].page_number == 1
    assert parsed.pages[0].char_start == 0
    assert parsed.pages[0].char_end == len(parsed.canonical_text)


@requires_pandoc
@pytest.mark.unit
def test_offset_fidelity_docx() -> None:
    """The Citation Engine precondition: every chunk slices back verbatim."""

    parsed = parse_docx(_plain_docx())
    chunks = chunk_document(parsed)

    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert (
            parsed.canonical_text[chunk.char_offset_start : chunk.char_offset_end] == chunk.content
        )


@requires_pandoc
@pytest.mark.unit
def test_redline_canonical_is_changes_accepted_text() -> None:
    """ADR 0017 §2: canonical text keeps insertions, drops deletions."""

    parsed = parse_docx(_redlined_docx())

    assert "USD 200" in parsed.canonical_text  # insertion kept
    assert "USD 100" not in parsed.canonical_text  # deletion dropped


@requires_pandoc
@pytest.mark.unit
def test_redline_reconstruction_matches_separate_accept_pass() -> None:
    """The spike-validated invariant: our reconstruction from the single
    ``--track-changes=all`` pass is byte-identical to what Pandoc itself
    produces with ``--track-changes=accept``."""

    import subprocess

    parsed = parse_docx(_redlined_docx())
    accept = subprocess.run(
        [
            shutil.which("pandoc"),
            "-f",
            "docx",
            "-t",
            "markdown",
            "--track-changes=accept",
            "--wrap=none",
            "--markdown-headings=atx",
            "--sandbox",
        ],
        input=_redlined_docx(),
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")

    assert parsed.canonical_text == accept


@requires_pandoc
@pytest.mark.unit
def test_redline_revision_layer_retains_author_and_date() -> None:
    parsed = parse_docx(_redlined_docx())

    assert parsed.structured_content is not None
    revisions = parsed.structured_content["revisions"]
    kinds = {r["kind"] for r in revisions}
    assert "insertion" in kinds
    assert "deletion" in kinds

    insertion = next(r for r in revisions if r["kind"] == "insertion")
    assert insertion["text"] == "USD 200"
    assert insertion["author"] == "Laura Fernández"
    assert insertion["date"].startswith("2026-01-02")
    # The anchor points into canonical_text and slices back verbatim.
    assert (
        parsed.canonical_text[insertion["char_start"] : insertion["char_end"]] == insertion["text"]
    )

    deletion = next(r for r in revisions if r["kind"] == "deletion")
    assert deletion["text"] == "USD 100"
    assert deletion["author"] == "Laura Fernández"
    # Deleted text is NOT in the canonical stream — its anchor is the point
    # where it would have been (an empty span).
    assert deletion["char_start"] == deletion["char_end"]


@requires_pandoc
@pytest.mark.unit
def test_paragraph_level_redline_matches_accept_pass() -> None:
    """Whole-paragraph insert/delete: canonical must equal Pandoc's own
    accept pass byte-for-byte (deleted paragraph vanishes with its block
    separator), and both revisions are retained."""

    import subprocess

    raw = _paragraph_redlined_docx()
    parsed = parse_docx(raw)
    accept = subprocess.run(
        [
            shutil.which("pandoc"),
            "-f",
            "docx",
            "-t",
            "markdown",
            "--track-changes=accept",
            "--wrap=none",
            "--markdown-headings=atx",
            "--sandbox",
        ],
        input=raw,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")

    assert parsed.canonical_text == accept
    assert "Cláusula nueva entera." in parsed.canonical_text
    assert "Cláusula borrada entera." not in parsed.canonical_text

    revisions = parsed.structured_content["revisions"]
    deletion = next(r for r in revisions if r["kind"] == "deletion")
    assert deletion["text"] == "Cláusula borrada entera."
    insertion = next(r for r in revisions if r["kind"] == "insertion")
    assert (
        parsed.canonical_text[insertion["char_start"] : insertion["char_end"]] == insertion["text"]
    )


@requires_pandoc
@pytest.mark.unit
def test_comment_retained_with_author_and_anchor() -> None:
    parsed = parse_docx(_commented_docx())

    # The commented body text stays in the canonical stream.
    assert "sin causa" in parsed.canonical_text

    assert parsed.structured_content is not None
    comments = [r for r in parsed.structured_content["revisions"] if r["kind"] == "comment"]
    assert len(comments) == 1
    comment = comments[0]
    assert comment["author"] == "Martín Pérez"
    assert "Pedir causa justificada" in comment["text"]
    # The anchor covers the commented span of canonical text.
    anchored = parsed.canonical_text[comment["char_start"] : comment["char_end"]]
    assert "sin causa" in anchored


@requires_pandoc
@pytest.mark.unit
def test_deterministic_for_same_bytes() -> None:
    raw = _redlined_docx()
    first = parse_docx(raw)
    second = parse_docx(raw)
    assert first.canonical_text == second.canonical_text
    assert first.structured_content == second.structured_content


@requires_pandoc
@pytest.mark.unit
def test_non_docx_bytes_raise_decode_error() -> None:
    with pytest.raises(ParserDecodeError):
        parse_docx(b"%PDF-1.7 definitely not a zip archive")


@requires_pandoc
@pytest.mark.unit
def test_zip_but_not_docx_raises_decode_error() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", "not a word document")
    with pytest.raises(ParserDecodeError):
        parse_docx(buf.getvalue())


@pytest.mark.unit
def test_empty_input_raises_parser_error() -> None:
    with pytest.raises(ParserError):
        parse_docx(b"")
