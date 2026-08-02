"""Tests for the Docling enrichment arq job and its ingest hook.

The job wrapper is best-effort and timeout-bounded; the decision of
whether ingest should chain it is a pure helper so it's testable
without spinning up the full ingest pipeline.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.workers.document_pipeline import (
    _should_enqueue_enrich,
    docling_enrich_job,
)


@pytest.mark.unit
def test_should_enqueue_enrich_when_ready_pymupdf_and_enabled() -> None:
    assert _should_enqueue_enrich(
        status="ready", parser="pymupdf", enrich_enabled=True
    )


@pytest.mark.unit
def test_should_not_enqueue_when_disabled() -> None:
    assert not _should_enqueue_enrich(
        status="ready", parser="pymupdf", enrich_enabled=False
    )


@pytest.mark.unit
def test_should_not_enqueue_for_docx() -> None:
    assert not _should_enqueue_enrich(
        status="ready", parser="pandoc", enrich_enabled=True
    )


@pytest.mark.unit
def test_should_not_enqueue_when_not_ready() -> None:
    assert not _should_enqueue_enrich(
        status="failed", parser="pymupdf", enrich_enabled=True
    )


@pytest.mark.unit
async def test_docling_enrich_job_times_out_cleanly() -> None:
    """A Docling pass that overruns the budget returns a failed result,
    never raises (arq must not retry a multi-minute OCR forever)."""

    async def slow_enrich(*args, **kwargs):
        await asyncio.sleep(5)

    fake_settings = type(
        "S", (), {"lq_ai_docling_timeout_seconds": 0.05}
    )()

    with (
        patch("app.workers.document_pipeline.get_settings", return_value=fake_settings),
        patch(
            "app.workers.document_pipeline._load_file_bytes",
            AsyncMock(return_value=b"%PDF fake"),
        ),
        patch("app.workers.document_pipeline.get_session_factory") as fake_factory,
        patch(
            "app.workers.document_pipeline.enrich_document_for_file",
            slow_enrich,
        ),
    ):
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=AsyncMock())
        session_cm.__aexit__ = AsyncMock(return_value=False)
        fake_factory.return_value = lambda: session_cm

        result = await docling_enrich_job(
            {"redis": None}, "11111111-1111-1111-1111-111111111111"
        )

    assert result["status"] == "failed"
    assert result["error"] == "timeout"
