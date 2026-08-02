"""Document parser adapters — PyMuPDF/Docling for PDF, Pandoc for DOCX, a stdlib decode for text.

Plain text and Markdown (:func:`parse_text`) need no parsing library: the
decoded bytes are themselves the canonical character stream, stored verbatim
so citations resolve byte-for-byte. DOCX (:func:`parse_docx`, ADR 0017) shells
out to a pinned Pandoc binary — the Markdown it emits is the canonical stream,
with tracked changes resolved to the *changes-accepted* text and the full
revision layer (insertions/deletions/comments with author + date) retained in
``structured_content``. PDF needs real extraction, per ADR 0006:

* **PyMuPDF** is the source of truth for the canonical character
  stream. Every chunk's ``content`` slices the PyMuPDF output by
  ``[char_offset_start:char_offset_end]`` byte-for-byte.
* **Docling** produces a structured representation (titles,
  paragraphs, tables) that is stashed for M2 consumption. M1's
  chunker does not consume Docling's offsets — they are not
  character-precise against the original PDF.

This module exposes a single high-level entry point :func:`parse_pdf`
that runs the cascade: PyMuPDF first (mandatory — without it we
can't produce offsets), Docling second (optional — failures
degrade gracefully). The returned :class:`ParsedDocument` carries
the canonical text, page boundaries, and Docling's structured
output (or ``None`` on Docling failure).

Both PyMuPDF and Docling are sync libraries. The orchestrator runs
them via :func:`asyncio.to_thread` so the worker event-loop is not
blocked. Imports are deferred until first call so the module
imports cleanly in environments where the libraries aren't
installed (e.g. CI runners without the worker dependencies).

Library versions are pinned via :mod:`api/pyproject.toml`. The
``parser_version`` returned by this module records the actually-loaded
library version at ingest time so re-ingest decisions can be made
against version drift.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ParserError(Exception):
    """Base class for parser errors raised inside the pipeline.

    Distinct from :class:`app.errors.LQAIError` because parser errors
    are internal to the pipeline — the orchestrator translates them
    into ``files.ingestion_error`` strings and ``ingestion_status =
    'failed'`` rather than HTTP responses.
    """


class ParserUnsupported(ParserError):
    """The file type or content is not supported by any installed parser.

    Currently raised for non-PDF MIME types (DOCX, RTF — M2) and for
    encrypted PDFs (the M1 pipeline does not unlock them).
    """


class ParserDecodeError(ParserError):
    """A text/markdown upload could not be decoded as UTF-8.

    Distinct from :class:`ParserUnsupported`: the MIME *is* a supported
    text type, but the bytes are not valid UTF-8. We fail loud rather
    than guess an encoding — a silent mis-decode would corrupt the
    canonical text a citation later verifies against. The orchestrator
    maps this to ``ingestion_error='decode_error'``.
    """


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PageSpan:
    """A page's span within the canonical character stream.

    Half-open interval ``[start, end)`` — slicing
    ``canonical_text[start:end]`` returns the page's text.
    """

    page_number: int  # 1-based, matches PDF page numbering convention
    char_start: int
    char_end: int


@dataclass(slots=True)
class ParsedDocument:
    """The canonical view of a parsed PDF.

    Attributes:
        canonical_text: The full, concatenated, character-precise text
            of the PDF as produced by PyMuPDF. This is the load-bearing
            artifact: every chunk's ``[char_offset_start:char_offset_end]``
            slice of this string equals the chunk's ``content``.
        pages: One :class:`PageSpan` per page; ``pages[i].page_number``
            is 1-based.
        page_count: Total page count; equals ``len(pages)``.
        parser: Which parser cascade produced this result —
            ``'docling+pymupdf'`` (both succeeded), ``'pymupdf'``
            (Docling fell through), or ``'pymupdf-only'``
            (Docling not attempted, e.g., disabled by config).
        parser_version: Library version string of the canonical
            parser (``fitz.__doc__`` for PyMuPDF, plus Docling
            version when applicable).
        structured_content: Docling's structured representation —
            ``None`` if Docling failed or wasn't attempted. M1 stashes
            this in ``documents.structured_content`` for M2 consumption;
            the chunker does not consume it.
    """

    canonical_text: str
    pages: list[PageSpan]
    page_count: int
    parser: str
    parser_version: str
    structured_content: dict[str, object] | None = field(default=None)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Sentinel MIME types we accept for PDFs. DOCX/RTF still stay at
# ``ingestion_status='failed'`` with ``unsupported_type``; plain
# text / Markdown are handled by the :func:`parse_text` branch below.
SUPPORTED_PDF_MIMES = frozenset(
    {
        "application/pdf",
        "application/x-pdf",
        "application/acrobat",
        "applications/vnd.pdf",
        "text/pdf",
        "text/x-pdf",
    }
)

# Plain-text and Markdown uploads. These need no parsing library — the
# decoded bytes are themselves the canonical character stream — so they
# take the :func:`parse_text` branch rather than the PyMuPDF/Docling
# cascade.
SUPPORTED_TEXT_MIMES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/x-markdown",
    }
)


def is_pdf_mime(mime_type: str) -> bool:
    """Return True if the MIME indicates a PDF the pipeline can handle.

    Some uploaders emit non-canonical MIME strings; we accept the
    common variants.
    """

    return mime_type.lower() in SUPPORTED_PDF_MIMES


def is_text_mime(mime_type: str) -> bool:
    """Return True if the MIME indicates plain text or Markdown.

    Text uploads commonly carry a charset parameter
    (``text/plain; charset=utf-8``), so we match on the bare type and
    ignore parameters. The charset is *not* honoured for decoding —
    :func:`parse_text` always decodes strict UTF-8 (see its docstring).
    """

    base = mime_type.split(";", 1)[0].strip().lower()
    return base in SUPPORTED_TEXT_MIMES


# Filename extensions treated as text when the browser-supplied MIME is
# unreliable. Browsers and operating systems disagree on the MIME for ``.md``
# (frequently ``application/octet-stream`` or empty), so the extension is the
# dependable signal. This only *routes* the upload to :func:`parse_text`, which
# still validates the bytes (strict UTF-8, no NUL) — so a binary file with a
# ``.md`` name fails cleanly as ``decode_error`` rather than being trusted.
TEXT_EXTENSIONS = (".md", ".markdown", ".txt")


def is_text_filename(filename: str) -> bool:
    """Return True if the filename has a known text/Markdown extension.

    The fallback for generic-MIME uploads (e.g. ``application/octet-stream``),
    which is what many browsers send for ``.md``.
    """

    return filename.lower().endswith(TEXT_EXTENSIONS)


def parse_text(raw_bytes: bytes) -> ParsedDocument:
    """Build a canonical :class:`ParsedDocument` from a text/Markdown upload.

    Unlike PDF, plain text needs no parsing library: the decoded bytes
    *are* the canonical character stream. Storing them verbatim makes the
    offset-fidelity contract
    (``canonical_text[char_offset_start:char_offset_end] == chunk.content``)
    hold exactly, so the Citation Engine re-reads byte-for-byte and the
    strongest (exact-match) citation tier resolves for every text document.

    Markdown is stored **verbatim, never rendered** — a citation must
    resolve against the source the user uploaded, not a derived rendering,
    and rendering would also add a non-deterministic dependency.

    Decoding is strict UTF-8 (``utf-8-sig`` transparently drops an optional
    BOM). A non-UTF-8 byte stream raises :class:`ParserDecodeError` rather
    than guessing an encoding — a silent mis-decode would corrupt the text a
    citation later verifies against. This is a pure, deterministic function:
    same bytes in, same canonical text out, with no model, OCR, or network.

    A NUL byte (``0x00``) is also rejected: it decodes as valid Unicode but
    PostgreSQL ``text`` columns cannot store it, so ``normalized_content``
    would fail to persist downstream. A NUL almost always means the upload is
    binary (or UTF-16) mislabeled as text — failing loud here is correct, and
    keeps the failure a clean ``decode_error`` rather than a DB exception that
    strands the row mid-ingest.
    """

    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParserDecodeError(
            f"upload is not valid UTF-8 text (offset {exc.start}): {exc.reason}. "
            "Save the file as UTF-8 (not UTF-16 / ANSI / Latin-1) and retry."
        ) from exc

    if "\x00" in text:
        raise ParserDecodeError(
            "upload contains a NUL byte (0x00); this is not a valid text "
            "document (likely a binary or UTF-16 file mislabeled as text)."
        )

    # Single synthetic page over the whole stream — text has no pagination;
    # this mirrors the autonomous-artifact construction in
    # ``app.autonomous.guard`` and keeps the chunker's page lookup happy.
    return ParsedDocument(
        canonical_text=text,
        pages=[PageSpan(page_number=1, char_start=0, char_end=len(text))],
        page_count=1,
        parser="plain-text",
        parser_version="1",
        structured_content=None,
    )


# ---------------------------------------------------------------------------
# DOCX (ADR 0017 — Pandoc canonical stream, tracked changes retained)
# ---------------------------------------------------------------------------

# The one MIME registered for OOXML WordprocessingML. Legacy binary .doc
# (application/msword) stays unsupported — OOXML only, per the mini-PRD
# scope cut.
SUPPORTED_DOCX_MIMES = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
)

# Extension fallback for generic-MIME uploads, mirroring TEXT_EXTENSIONS:
# browsers occasionally send .docx as application/octet-stream. Routing by
# extension is safe because parse_docx still validates the bytes (Pandoc
# rejects a non-OOXML package), so a mislabeled file fails cleanly.
DOCX_EXTENSIONS = (".docx",)

# Wall-clock budget for the Pandoc subprocess (ADR 0017 §5 — untrusted
# input must never hang the worker). Generous: real contracts convert in
# well under a second; a pathological package hits this and fails loud.
PANDOC_TIMEOUT_SECONDS = 120

_PANDOC_ARGS = (
    "-f",
    "docx",
    "-t",
    "markdown",
    "--track-changes=all",
    "--wrap=none",
    "--markdown-headings=atx",
    "--sandbox",
)

# Track-change / comment spans in Pandoc's markdown output, e.g.
#   [USD 200]{.insertion author="Laura" date="2026-01-02T10:00:00Z"}
# The bracketed text may contain backslash-escaped brackets.
_SPAN_RX = None  # compiled lazily in _resolve_track_changes


def is_docx_mime(mime_type: str) -> bool:
    """Return True if the MIME indicates an OOXML Word document.

    Parameters (rare on this type, but uploaders vary) are ignored, matching
    :func:`is_text_mime`.
    """

    base = mime_type.split(";", 1)[0].strip().lower()
    return base in SUPPORTED_DOCX_MIMES


def is_docx_filename(filename: str) -> bool:
    """Return True if the filename has the .docx extension.

    The fallback for generic-MIME uploads (application/octet-stream), same
    role as :func:`is_text_filename` for text.
    """

    return filename.lower().endswith(DOCX_EXTENSIONS)


def parse_docx(raw_bytes: bytes) -> ParsedDocument:
    """Build a canonical :class:`ParsedDocument` from a .docx upload (ADR 0017).

    One ``pandoc --track-changes=all`` pass produces Markdown carrying the
    tracked changes and comments inline as spans. From it we derive:

    * ``canonical_text`` — the *changes-accepted* text (insertions kept,
      deletions dropped, comment markers removed). Byte-identical to a
      separate ``--track-changes=accept`` pass (spike-validated; enforced
      by test), so the offset-fidelity contract holds and the Citation
      Engine re-reads verbatim. DOCX has no fixed pagination, so the
      document is one synthetic page (same convention as ``parse_text``).
    * ``structured_content["revisions"]`` — every insertion, deletion, and
      comment with ``text``, ``author``, ``date``, and a char anchor into
      ``canonical_text`` (deletions anchor as empty spans at the point the
      text was removed). Retained, not discarded — the redline *is* the
      signal in legal review.

    Pandoc is invoked as a separate subprocess at arm's length (GPL posture,
    ADR 0017 §6) with ``--sandbox`` and a wall-clock timeout (§5). Known v1
    gaps, per the ADR: a comment anchored to an unaccepted insertion is
    dropped by Pandoc itself (upstream #9833 — the OOXML fallback is future
    work), and comment reply threads are not reconstructed.
    """

    import shutil
    import subprocess

    if not raw_bytes:
        raise ParserError("DOCX input is empty")

    binary = shutil.which("pandoc")
    if binary is None:
        raise ParserError(
            "pandoc binary not found on PATH; DOCX ingestion cannot run "
            "(ADR 0017 pins it in the api and ingest-worker images)"
        )

    try:
        proc = subprocess.run(
            [binary, *_PANDOC_ARGS],
            input=raw_bytes,
            capture_output=True,
            timeout=PANDOC_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ParserError(f"pandoc exceeded {PANDOC_TIMEOUT_SECONDS}s parsing the DOCX") from exc

    if proc.returncode != 0:
        # Pandoc's stderr on bad input names the failure without echoing
        # document content ("couldn't unpack docx container: …").
        detail = proc.stderr.decode("utf-8", errors="replace").strip()[:200]
        raise ParserDecodeError(f"not a readable .docx package: {detail}")

    markdown_all = proc.stdout.decode("utf-8")
    canonical_text, revisions = _resolve_track_changes(markdown_all)

    if "\x00" in canonical_text:
        raise ParserDecodeError(
            "extracted DOCX text contains a NUL byte (0x00); refusing to "
            "persist (PostgreSQL text columns cannot store it)"
        )

    return ParsedDocument(
        canonical_text=canonical_text,
        pages=[PageSpan(page_number=1, char_start=0, char_end=len(canonical_text))],
        page_count=1,
        parser="pandoc",
        parser_version=f"pandoc={_safe_pandoc_version(binary)}",
        structured_content={
            "source": "pandoc-track-changes",
            "revisions": revisions,
        },
    )


def _resolve_track_changes(
    markdown_all: str,
) -> tuple[str, list[dict[str, object]]]:
    """Resolve ``--track-changes=all`` spans into accepted text + revisions.

    Walks the Markdown once. Insertion spans unwrap into the output;
    deletion spans are dropped (recorded as empty-span anchors); comment
    start/end markers are removed while the commented body text stays.
    Spans with any other class (e.g. ``{.underline}``) pass through
    verbatim — they are part of the canonical stream.

    A fully-deleted paragraph leaves an empty block behind; Pandoc's own
    ``accept`` output removes the block *and* its blank-line separator, so
    :func:`_collapse_empty_blocks` normalises the newline runs afterwards
    (adjusting the recorded anchors) to preserve byte-identity.
    """

    import re

    global _SPAN_RX
    if _SPAN_RX is None:
        _SPAN_RX = re.compile(r"\[((?:\\.|[^\]\\])*)\]\{([^}]*)\}")

    out: list[str] = []
    out_len = 0
    revisions: list[dict[str, object]] = []
    open_comments: list[dict[str, object]] = []
    pos = 0

    def _attr(attrs: str, name: str) -> str:
        m = re.search(rf'{name}="([^"]*)"', attrs)
        return m.group(1) if m else ""

    for match in _SPAN_RX.finditer(markdown_all):
        literal = markdown_all[pos : match.start()]
        out.append(literal)
        out_len += len(literal)
        pos = match.end()

        inner, attrs = match.group(1), match.group(2)
        kind = attrs.split(None, 1)[0] if attrs else ""

        if kind == ".insertion":
            revisions.append(
                {
                    "kind": "insertion",
                    "text": inner,
                    "author": _attr(attrs, "author"),
                    "date": _attr(attrs, "date"),
                    "char_start": out_len,
                    "char_end": out_len + len(inner),
                }
            )
            out.append(inner)
            out_len += len(inner)
        elif kind == ".deletion":
            revisions.append(
                {
                    "kind": "deletion",
                    "text": inner,
                    "author": _attr(attrs, "author"),
                    "date": _attr(attrs, "date"),
                    "char_start": out_len,
                    "char_end": out_len,
                }
            )
        elif kind == ".comment-start":
            # The bracketed text of a comment-start span is the comment
            # body; the *anchored document text* follows until the paired
            # comment-end marker. The body is not document text — record
            # it, emit nothing.
            open_comments.append(
                {
                    "kind": "comment",
                    "id": _attr(attrs, "id"),
                    "text": inner,
                    "author": _attr(attrs, "author"),
                    "date": _attr(attrs, "date"),
                    "char_start": out_len,
                }
            )
        elif kind == ".comment-end":
            comment_id = _attr(attrs, "id")
            idx = next(
                (i for i, c in enumerate(open_comments) if c["id"] == comment_id),
                len(open_comments) - 1,
            )
            if idx >= 0:
                comment = open_comments.pop(idx)
                comment.pop("id", None)
                comment["char_end"] = out_len
                revisions.append(comment)
        elif kind in (".paragraph-insertion", ".paragraph-deletion"):
            # Paragraph-mark change markers: always empty spans; the run
            # content is carried by the sibling .insertion/.deletion span.
            pass
        else:
            # Not a track-change span — canonical Markdown, keep verbatim.
            whole = match.group(0)
            out.append(whole)
            out_len += len(whole)

    tail = markdown_all[pos:]
    out.append(tail)
    out_len += len(tail)

    # Defensive: an unpaired comment-start closes at end-of-document.
    for comment in open_comments:
        comment.pop("id", None)
        comment["char_end"] = out_len
        revisions.append(comment)

    text = "".join(out)
    text, revisions = _collapse_empty_blocks(text, revisions)
    revisions.sort(key=lambda r: (r["char_start"], r["char_end"]))
    return text, revisions


def _collapse_empty_blocks(
    text: str, revisions: list[dict[str, object]]
) -> tuple[str, list[dict[str, object]]]:
    """Normalise newline runs left behind by fully-deleted paragraphs.

    Pandoc's Markdown separates blocks with exactly one blank line and
    never emits three-plus consecutive newlines — so any such run in the
    resolved text is the residue of a dropped (fully-deleted) block.
    Collapse interior runs to ``\\n\\n``, strip a leading run, reduce a
    trailing run to a single ``\\n``, and shift the revision anchors that
    sit at or beyond each collapse point so they keep slicing verbatim.
    """

    import re

    replacements: list[tuple[int, int, str]] = []
    lead = re.match(r"\n+", text)
    if lead:
        replacements.append((0, lead.end(), ""))
    for m in re.finditer(r"\n{3,}", text):
        if lead and m.start() < lead.end():
            continue
        replacement = "\n" if m.end() == len(text) else "\n\n"
        replacements.append((m.start(), m.end(), replacement))

    if not replacements:
        return text, revisions

    def _shift(offset: int) -> int:
        shifted = offset
        for start, end, repl in replacements:
            if offset >= end:
                shifted -= (end - start) - len(repl)
            elif offset > start:
                # Anchor inside a collapsed run (a deletion's point anchor):
                # snap it to just after the replacement separator.
                shifted -= offset - start - min(offset - start, len(repl))
        return shifted

    from typing import cast

    for rev in revisions:
        rev["char_start"] = _shift(cast(int, rev["char_start"]))
        rev["char_end"] = _shift(cast(int, rev["char_end"]))

    pieces: list[str] = []
    cursor = 0
    for start, end, repl in replacements:
        pieces.append(text[cursor:start])
        pieces.append(repl)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), revisions


def _safe_pandoc_version(binary: str) -> str:
    """Return the pinned Pandoc binary's version string, defensively.

    Cached per-process — the binary is pinned in the image (ADR 0017 §4),
    so it cannot change under a running worker.
    """

    global _PANDOC_VERSION
    if _PANDOC_VERSION is None:
        import re
        import subprocess

        try:
            head = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                timeout=10,
                check=True,
            ).stdout.decode("utf-8", errors="replace")
            m = re.match(r"pandoc(?:\.exe)?\s+([\w.]+)", head)
            _PANDOC_VERSION = m.group(1) if m else "unknown"
        except Exception:  # pragma: no cover — defensive
            _PANDOC_VERSION = "unknown"
    return _PANDOC_VERSION


_PANDOC_VERSION: str | None = None


def parse_pdf(pdf_bytes: bytes, *, run_docling: bool = True) -> ParsedDocument:
    """Run the parser cascade on a PDF byte string.

    PyMuPDF is run first and is mandatory — without it we cannot
    produce the canonical character stream. If PyMuPDF raises, this
    function re-raises as :class:`ParserError`.

    Docling is run second and is optional; on Docling failure we log
    a WARNING and proceed with PyMuPDF-only results.

    The function is sync so the worker can wrap it via
    :func:`asyncio.to_thread`. Both libraries are sync internally.

    Args:
        pdf_bytes: Raw PDF byte string. Empty input raises
            :class:`ParserError`.
        run_docling: When False, skip the Docling pass entirely.
            Useful in tests where Docling is mocked or unavailable.

    Returns:
        :class:`ParsedDocument` with the canonical text, page spans,
        parser metadata, and Docling's structured content if
        successful.
    """

    if not pdf_bytes:
        raise ParserError("PDF input is empty")

    # PyMuPDF is the canonical parser — without it, no offsets, no
    # ingestion. We import it lazily so this module imports cleanly
    # in environments where it isn't installed (test stubs, etc.).
    canonical_text, pages, pymupdf_version = _run_pymupdf(pdf_bytes)

    # Docling is best-effort. Skip if disabled or unavailable.
    structured_content: dict[str, object] | None = None
    docling_version: str | None = None
    docling_succeeded = False

    if run_docling:
        try:
            structured_content, docling_version, _ = _run_docling(pdf_bytes)
            docling_succeeded = True
        except Exception as exc:
            # Docling failures are recoverable — we degrade to
            # PyMuPDF-only and log so operators see the drift.
            log.warning(
                "Docling parser failed; falling back to PyMuPDF-only",
                extra={
                    "event": "pipeline_docling_fallback",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

    if docling_succeeded:
        parser_label = "docling+pymupdf"
        version_label = f"pymupdf={pymupdf_version}; docling={docling_version}"
    elif run_docling:
        parser_label = "pymupdf"
        version_label = f"pymupdf={pymupdf_version}; docling=fallback"
    else:
        parser_label = "pymupdf-only"
        version_label = f"pymupdf={pymupdf_version}"

    return ParsedDocument(
        canonical_text=canonical_text,
        pages=pages,
        page_count=len(pages),
        parser=parser_label,
        parser_version=version_label,
        structured_content=structured_content,
    )


# ---------------------------------------------------------------------------
# PyMuPDF adapter
# ---------------------------------------------------------------------------


def _run_pymupdf(pdf_bytes: bytes) -> tuple[str, list[PageSpan], str]:
    """Extract canonical text + page spans + library version via PyMuPDF.

    The canonical text is built by concatenating every page's
    extracted text in document order. Page boundaries are recorded as
    offsets into this concatenated string so the chunker (and M2's
    citation engine) can map an offset back to a page.

    PyMuPDF's ``page.get_text()`` returns a string per page. We use
    this directly: the canonical text is exactly what PyMuPDF says it
    is, with the same character ordering and the same byte content.
    Slicing the canonical text by ``[start:end]`` is therefore
    equivalent to slicing the per-page output PyMuPDF returns —
    that's what makes the offsets character-precise.

    Page joining: pages are joined with ``\n`` (a single newline). The
    chunker treats the newline as a regular character; offsets count
    it like any other.

    Raises:
        :class:`ParserError`: PyMuPDF failed to open or read the PDF
            (corrupt file, encrypted document we can't unlock, etc.).
    """

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover — install-time error
        raise ParserError("PyMuPDF (fitz) is not installed; document pipeline cannot run") from exc

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ParserError(f"PyMuPDF could not open the PDF: {exc}") from exc

    try:
        if doc.is_encrypted:
            raise ParserUnsupported(
                "Encrypted PDFs are not supported in M1 (no OCR / no decryption)"
            )

        page_texts: list[str] = []
        pages: list[PageSpan] = []
        running_offset = 0
        page_count = doc.page_count

        for page_idx in range(page_count):
            page = doc.load_page(page_idx)
            try:
                text = page.get_text()
            except Exception as exc:
                raise ParserError(
                    f"PyMuPDF failed extracting text from page {page_idx + 1}: {exc}"
                ) from exc

            # Track this page's span before appending the join character.
            page_span = PageSpan(
                page_number=page_idx + 1,
                char_start=running_offset,
                char_end=running_offset + len(text),
            )
            pages.append(page_span)
            page_texts.append(text)
            running_offset += len(text)

            # Add a single newline between pages — except after the last.
            if page_idx < page_count - 1:
                running_offset += 1  # account for the join newline

        canonical_text = "\n".join(page_texts)

        # Sanity: the running offset must match the canonical text length
        # exactly. If it doesn't, our offset-tracking has a bug and the
        # chunker downstream will produce wrong slices.
        if running_offset != len(canonical_text):  # pragma: no cover — defensive
            raise ParserError(
                "PyMuPDF offset accounting drift: "
                f"running_offset={running_offset}, "
                f"canonical_text_len={len(canonical_text)}"
            )

        version = _safe_pymupdf_version()
        return canonical_text, pages, version
    finally:
        with contextlib.suppress(Exception):
            doc.close()  # pragma: no cover — closing is best-effort


def _safe_pymupdf_version() -> str:
    """Return the loaded PyMuPDF version string, defensively.

    PyMuPDF exposes ``fitz.version`` as a tuple in newer releases and
    ``fitz.__doc__`` in older ones; fall back to ``"unknown"`` if the
    attribute isn't where we expect.
    """

    try:
        import fitz

        version = getattr(fitz, "version", None)
        if version is not None:
            return str(version[0]) if isinstance(version, tuple) else str(version)
        return getattr(fitz, "__version__", "unknown")
    except Exception:  # pragma: no cover — defensive
        return "unknown"


# ---------------------------------------------------------------------------
# Docling adapter
# ---------------------------------------------------------------------------


def _run_docling(pdf_bytes: bytes, *, do_ocr: bool = False) -> tuple[dict[str, object], str, str]:
    """Run Docling against the PDF; return (structured, version, text).

    Docling's API surface (as of v1.x) accepts an in-memory document
    via its converter. We use the ``DocumentConverter`` entry point
    and store the produced document's serialised ``model_dump()`` so
    M2 readers can deserialise it back into Docling objects.

    ``do_ocr=False`` (the default) matters: Docling 1.x's own default
    is OCR-on with four EasyOCR languages, which costs minutes per PDF
    on CPU even for born-digital documents. The enrichment job opts in
    only for image-only PDFs, restricted to es+en.

    ``text`` is Docling's plain-text export, populated only on the OCR
    path (it's the OCR'd character stream the caller re-chunks); empty
    string otherwise.

    Raises:
        Any exception Docling raises. Caller catches and falls back to
        PyMuPDF-only.
    """

    try:
        from docling.datamodel.base_models import DocumentStream
        from docling.datamodel.pipeline_options import EasyOcrOptions, PipelineOptions
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ParserError(
            "Docling is not installed; document pipeline cannot run Docling pass"
        ) from exc

    import io

    pipeline_options = PipelineOptions(do_ocr=do_ocr)
    if do_ocr:
        pipeline_options.ocr_options = EasyOcrOptions(lang=["es", "en"], use_gpu=False)
    converter = DocumentConverter(pipeline_options=pipeline_options)
    stream = DocumentStream(name="upload.pdf", stream=io.BytesIO(pdf_bytes))
    result = converter.convert(stream)

    # Newer Docling exposes the result on .document; older on .output.
    doc = getattr(result, "document", None) or getattr(result, "output", None)
    if doc is None:
        raise ParserError("Docling returned no document on conversion result")

    # Serialise via model_dump if Pydantic-backed; otherwise best-effort
    # (older Docling versions return a custom object — coerce to dict).
    structured = doc.model_dump() if hasattr(doc, "model_dump") else {"raw": str(doc)}

    version = _safe_docling_version()
    text = _docling_text(doc) if do_ocr else ""
    return structured, version, text


def _docling_text(doc: object) -> str:
    """Best-effort plain text from a Docling document (for the OCR path)."""

    for attr in ("export_to_markdown", "export_to_text"):
        fn = getattr(doc, attr, None)
        if callable(fn):
            return str(fn())
    return ""


def _safe_docling_version() -> str:
    """Return Docling's installed version, defensively."""

    try:
        import docling

        return getattr(docling, "__version__", "unknown")
    except Exception:  # pragma: no cover — defensive
        return "unknown"
