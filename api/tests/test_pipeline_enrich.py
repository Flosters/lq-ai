"""Tests for the post-ready Docling enrichment (app/pipeline/enrich.py).

The enrichment is best-effort: it fills ``documents.structured_content``
after the fast path already made the file usable, and OCRs image-only
PDFs. It must never raise, never touch Pandoc's DOCX redline layer, and
never run when structure is already present.

The Docling call is injected (``docling_runner``) so these tests run
without importing real Docling — same isolation idea as embed.py's
injectable gateway.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk
from app.models.file import File as FileModel
from app.models.user import User
from app.pipeline.enrich import enrich_document_for_file
from app.security import hash_password


@pytest_asyncio.fixture
async def db_user(db_session: AsyncSession) -> User:
    user = User(
        email=f"enrich-{uuid.uuid4().hex[:8]}@example.com",
        display_name="Enrich Test User",
        hashed_password=hash_password("correct-horse-battery-staple"),
        is_admin=False,
        mfa_enabled=False,
        must_change_password=False,
    )
    db_session.add(user)
    await db_session.flush()
    return user


async def _make_file_with_document(
    db: AsyncSession,
    user: User,
    *,
    parser: str = "pymupdf",
    normalized_content: str = "texto del contrato",
    structured_content: dict | None = None,
) -> FileModel:
    file_row = FileModel(
        owner_id=user.id,
        filename="contrato.pdf",
        mime_type="application/pdf",
        size_bytes=100,
        hash_sha256="0" * 64,
        storage_path=str(uuid.uuid4()),
        ingestion_status="ready",
    )
    db.add(file_row)
    await db.flush()
    doc = Document(
        file_id=file_row.id,
        parser=parser,
        parser_version="pymupdf=test",
        page_count=1,
        character_count=len(normalized_content),
        structured_content=structured_content,
        normalized_content=normalized_content,
        was_ocrd=False,
    )
    db.add(doc)
    await db.flush()
    return file_row


async def _reload_document(db: AsyncSession, file_id: uuid.UUID) -> Document:
    return (
        await db.execute(select(Document).where(Document.file_id == file_id))
    ).scalar_one()


async def _load_chunks(db: AsyncSession, document_id: uuid.UUID) -> list[DocumentChunk]:
    rows = (
        await db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
    ).scalars()
    return list(rows)


@pytest.mark.integration
async def test_enrich_fills_structured_content(
    db_session: AsyncSession, db_user: User
) -> None:
    file_row = await _make_file_with_document(db_session, db_user)

    def fake_runner(pdf_bytes: bytes, *, do_ocr: bool):
        assert do_ocr is False
        return ({"pages": []}, "1.20.0-fake", "")

    result = await enrich_document_for_file(
        db_session, file_row.id, docling_runner=fake_runner, pdf_bytes=b"%PDF fake"
    )

    assert result.status == "enriched"
    doc = await _reload_document(db_session, file_row.id)
    assert doc.structured_content == {"pages": []}
    assert "docling=1.20.0-fake" in doc.parser_version


@pytest.mark.integration
async def test_enrich_skips_docx(db_session: AsyncSession, db_user: User) -> None:
    """Pandoc's redline layer (ADR 0017) must never be overwritten."""

    file_row = await _make_file_with_document(
        db_session, db_user, parser="pandoc", structured_content={"redlines": []}
    )

    result = await enrich_document_for_file(
        db_session, file_row.id, docling_runner=None, pdf_bytes=b"irrelevant"
    )

    assert result.status == "skipped"
    doc = await _reload_document(db_session, file_row.id)
    assert doc.structured_content == {"redlines": []}


@pytest.mark.integration
async def test_enrich_noop_when_structured_present(
    db_session: AsyncSession, db_user: User
) -> None:
    file_row = await _make_file_with_document(
        db_session, db_user, structured_content={"already": "there"}
    )

    result = await enrich_document_for_file(
        db_session, file_row.id, docling_runner=None, pdf_bytes=b"irrelevant"
    )

    assert result.status == "skipped"
    doc = await _reload_document(db_session, file_row.id)
    assert doc.structured_content == {"already": "there"}


@pytest.mark.integration
async def test_enrich_failure_leaves_document_untouched(
    db_session: AsyncSession, db_user: User
) -> None:
    file_row = await _make_file_with_document(db_session, db_user)

    def exploding_runner(pdf_bytes: bytes, *, do_ocr: bool):
        raise RuntimeError("docling exploded")

    result = await enrich_document_for_file(
        db_session, file_row.id, docling_runner=exploding_runner, pdf_bytes=b"%PDF fake"
    )

    assert result.status == "failed"
    assert result.error is not None and "docling exploded" in result.error
    doc = await _reload_document(db_session, file_row.id)
    assert doc.structured_content is None
    assert doc.parser_version == "pymupdf=test"
