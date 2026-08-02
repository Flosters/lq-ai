"""Tests for the Docling enrichment arq job and its ingest hook.

The job wrapper is best-effort and timeout-bounded; the decision of
whether ingest should chain it is a pure helper so it's testable
without spinning up the full ingest pipeline.
"""

from __future__ import annotations

import pytest

from app.workers.document_pipeline import _should_enqueue_enrich


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
