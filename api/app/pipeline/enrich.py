"""Post-ready Docling enrichment.

The fast ingest path (PyMuPDF + embeddings) makes a file usable in
seconds. This module runs *afterward*, as a background job, to fill
``documents.structured_content`` with Docling's structure and — for
image-only (scanned) PDFs that PyMuPDF could not extract text from — to
OCR the pages, re-chunk, and set ``was_ocrd=True`` so the document
becomes searchable and citable.

Contract:

* **Best-effort.** Any failure logs and returns ``status="failed"``;
  it never raises and never regresses what the fast path produced.
* **Never touches DOCX.** Pandoc stashes its tracked-changes/redline
  layer in ``structured_content`` (ADR 0017); enrichment only runs for
  PyMuPDF-parsed PDFs and never overwrites an existing structure.
* **Injected Docling.** ``docling_runner`` is injectable so tests run
  without importing Docling — the same isolation embed.py uses for the
  gateway. The default runner wraps ``parsers._run_docling``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Callable, Protocol

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.document import Document, DocumentChunk
from app.pipeline.chunker import chunk_document
from app.pipeline.parsers import PageSpan, ParsedDocument

log = logging.getLogger(__name__)


class DoclingRunner(Protocol):
    """Callable shape of a Docling pass: ``(pdf_bytes, *, do_ocr) -> (structured, version, text)``."""

    def __call__(
        self, pdf_bytes: bytes, *, do_ocr: bool
    ) -> tuple[dict[str, object], str, str]: ...


@dataclass(frozen=True)
class EnrichResult:
    file_id: uuid.UUID
    status: str  # enriched | ocr_enriched | skipped | failed
    error: str | None = None


def _default_runner(pdf_bytes: bytes, *, do_ocr: bool) -> tuple[dict[str, object], str, str]:
    from app.pipeline.parsers import _run_docling

    return _run_docling(pdf_bytes, do_ocr=do_ocr)


async def enrich_document_for_file(
    db: AsyncSession,
    file_id: uuid.UUID,
    *,
    pdf_bytes: bytes,
    docling_runner: Callable[..., tuple[dict[str, object], str, str]] | None = None,
) -> EnrichResult:
    """Enrich the document for ``file_id`` with Docling structure/OCR.

    Returns an :class:`EnrichResult`; never raises. ``pdf_bytes`` is the
    file's raw bytes (the caller loads them from storage).
    """

    runner = docling_runner or _default_runner

    doc = (
        await db.execute(select(Document).where(Document.file_id == file_id))
    ).scalar_one_or_none()

    if doc is None:
        return EnrichResult(file_id=file_id, status="skipped", error="no document")

    # Only PDFs parsed by PyMuPDF enroll. DOCX (Pandoc) owns
    # structured_content for its redline layer — never overwrite it.
    if not (doc.parser or "").startswith("pymupdf"):
        return EnrichResult(file_id=file_id, status="skipped", error="not a pymupdf pdf")

    # Idempotent: structure already present → nothing to do.
    if doc.structured_content is not None:
        return EnrichResult(file_id=file_id, status="skipped", error="already enriched")

    needs_ocr = not (doc.normalized_content or "").strip()

    # Docling is the only fallible step and it touches no DB state — run
    # it first, before any mutation, so a failure leaves the persisted
    # fast-path document exactly as it was (no rollback needed). The
    # caller owns the commit.
    try:
        structured, version, text = runner(pdf_bytes, do_ocr=needs_ocr)
    except Exception as exc:
        log.warning(
            "docling enrichment failed",
            extra={
                "event": "docling_enrich_failed",
                "file_id": str(file_id),
                "error": str(exc),
            },
        )
        return EnrichResult(file_id=file_id, status="failed", error=f"{type(exc).__name__}: {exc}")

    doc.structured_content = structured
    doc.parser_version = f"{doc.parser_version or ''}; docling={version} (enrich)"

    if needs_ocr and text.strip():
        status = await _apply_ocr_text(db, doc, text)
    else:
        await db.flush()
        status = "enriched"

    log.info(
        "docling enrichment complete",
        extra={"event": "docling_enrich_done", "file_id": str(file_id), "status": status},
    )
    return EnrichResult(file_id=file_id, status=status)


async def _apply_ocr_text(db: AsyncSession, doc: Document, text: str) -> str:
    """Replace an image-only document's empty text with OCR'd text and re-chunk.

    Preserves the Citation Engine invariant
    ``chunk.content == normalized_content[start:end]`` by chunking the
    exact string we store as ``normalized_content``.
    """

    settings = get_settings()

    doc.normalized_content = text
    doc.character_count = len(text)
    doc.was_ocrd = True

    parsed = ParsedDocument(
        canonical_text=text,
        pages=[PageSpan(page_number=1, char_start=0, char_end=len(text))],
        page_count=doc.page_count or 1,
        parser=doc.parser or "pymupdf",
        parser_version=doc.parser_version or "",
        structured_content=doc.structured_content,
    )
    chunks = chunk_document(
        parsed,
        target_chars=settings.lq_ai_chunk_target_chars,
        overlap_chars=settings.lq_ai_chunk_overlap_chars,
    )

    # Delete any prior chunks (an image-only doc has none, but the
    # (document_id, chunk_index) UNIQUE constraint makes this safe if a
    # partial run left some behind).
    await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
    await db.flush()

    for chunk in chunks:
        db.add(
            DocumentChunk(
                document_id=doc.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                char_offset_start=chunk.char_offset_start,
                char_offset_end=chunk.char_offset_end,
                tokens=None,
                metadata_json=chunk.metadata,
            )
        )
    await db.flush()
    return "ocr_enriched"
