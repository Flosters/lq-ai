"""Citation Engine — extraction tests.

The extractor parses the assistant response for quote-then-locate
citations: a double-quoted passage immediately followed by a
``(Source: [N])`` marker referring to a retrieved-chunk index.

Each successfully located citation is materialized as a
``CitationCandidate`` carrying the source_file_id / document_id /
byte-precise offsets / source_text / page — everything the verifier
and the persistence layer need.

Quotes that can't be located inside their cited chunk are dropped
silently for M2-A2. The schema requires file_id + offsets to be
non-null, and we don't speculate where an unfindable quote came from.
"Model claimed to cite but we can't find it" is a future failure-mode
audit task (DE candidate).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest

from app.citation.extraction import CitationCandidate, extract_citations


@dataclass(slots=True)
class _StubChunk:
    """HybridSearchResult-shaped stub for extractor tests — no DB needed."""

    document_id: uuid.UUID
    file_id: uuid.UUID
    content: str
    page_start: int | None
    char_offset_start: int
    char_offset_end: int


def _chunk(
    *,
    content: str,
    char_offset_start: int = 0,
    page_start: int | None = 1,
) -> _StubChunk:
    return _StubChunk(
        document_id=uuid.uuid4(),
        file_id=uuid.uuid4(),
        content=content,
        page_start=page_start,
        char_offset_start=char_offset_start,
        char_offset_end=char_offset_start + len(content),
    )


@pytest.mark.unit
def test_extract_single_citation_byte_precise_offsets() -> None:
    """A verbatim quote with (Source: [1]) becomes a candidate with derived offsets."""

    chunk = _chunk(
        content="The contract term shall be five years.",
        char_offset_start=200,
    )
    response = 'The agreement says "The contract term shall be five years." (Source: [1]).'

    candidates = extract_citations(response, [chunk])

    assert len(candidates) == 1
    cite = candidates[0]
    assert isinstance(cite, CitationCandidate)
    assert cite.source_file_id == chunk.file_id
    assert cite.source_document_id == chunk.document_id
    # Quote starts at the beginning of the chunk → offsets relative to the
    # document begin at chunk.char_offset_start.
    assert cite.source_offset_start == 200
    assert cite.source_offset_end == 200 + len("The contract term shall be five years.")
    assert cite.source_text == "The contract term shall be five years."
    assert cite.source_page == 1


@pytest.mark.unit
def test_extract_handles_offset_within_chunk() -> None:
    """Quote in the middle of a chunk → offsets account for the position inside."""

    # The chunk text places the quote at character 4 within the chunk.
    chunk = _chunk(content="==> The cited bit. ==>", char_offset_start=1000)
    response = 'It says "The cited bit." (Source: [1]).'

    candidates = extract_citations(response, [chunk])

    assert len(candidates) == 1
    cite = candidates[0]
    assert cite.source_offset_start == 1000 + chunk.content.find("The cited bit.")
    assert cite.source_offset_end == cite.source_offset_start + len("The cited bit.")
    assert cite.source_text == "The cited bit."


@pytest.mark.unit
def test_extract_multiple_citations_resolve_independent_chunks() -> None:
    """Two quotes citing two chunks produce two candidates."""

    chunk1 = _chunk(content="First fact statement.", char_offset_start=0)
    chunk2 = _chunk(content="Second fact assertion.", char_offset_start=500, page_start=3)
    response = (
        'He said "First fact statement." (Source: [1]) and '
        'also "Second fact assertion." (Source: [2]).'
    )

    candidates = extract_citations(response, [chunk1, chunk2])

    assert len(candidates) == 2
    assert candidates[0].source_file_id == chunk1.file_id
    assert candidates[1].source_file_id == chunk2.file_id
    assert candidates[1].source_page == 3


@pytest.mark.unit
def test_extract_drops_quote_without_source_marker() -> None:
    """A bare quote without `(Source: [N])` is not a citation; ignored."""

    chunk = _chunk(content="Some text.")
    response = 'He said "Some text." but cited nothing.'

    assert extract_citations(response, [chunk]) == []


@pytest.mark.unit
def test_extract_drops_source_marker_without_quote() -> None:
    """A standalone `(Source: [N])` with no preceding quote is ignored."""

    chunk = _chunk(content="Some text.")
    response = "Without quoting, the source is (Source: [1])."

    assert extract_citations(response, [chunk]) == []


@pytest.mark.unit
def test_extract_drops_unfindable_quote() -> None:
    """A quote not present in the cited chunk's content is dropped (M2-A2 policy)."""

    chunk = _chunk(content="The actual chunk text.")
    response = 'The model wrote "a fabricated quote." (Source: [1]).'

    assert extract_citations(response, [chunk]) == []


@pytest.mark.unit
def test_extract_drops_out_of_range_source_index() -> None:
    """`(Source: [99])` when only 2 chunks were retrieved is dropped."""

    chunk = _chunk(content="Real content.")
    response = 'He said "Real content." (Source: [99]).'

    assert extract_citations(response, [chunk]) == []


@pytest.mark.unit
def test_extract_tolerates_whitespace_around_source_marker() -> None:
    """Permissive: optional whitespace between quote and (Source: ...)."""

    chunk = _chunk(content="Cited text.")
    response = 'It says "Cited text."   (Source: [1]) here.'

    candidates = extract_citations(response, [chunk])
    assert len(candidates) == 1
    assert candidates[0].source_text == "Cited text."


@pytest.mark.unit
def test_extract_handles_smart_quotes() -> None:
    """M2-B1: extractor accepts curly-quote pairs; Stage 2 verifier will pass them."""

    chunk = _chunk(content="Cited text.")
    response = "It says “Cited text.” (Source: [1]) here."

    candidates = extract_citations(response, [chunk])
    assert len(candidates) == 1
    # source_text preserves the model's quote shape (smart quotes); the
    # verifier normalizes both sides before comparing.
    assert candidates[0].source_text == "Cited text."


@pytest.mark.unit
def test_extract_handles_multiline_quote() -> None:
    """Quotes spanning newlines are extracted (regex must cross line breaks)."""

    chunk_content = "First line.\nSecond line of the same quote."
    chunk = _chunk(content=chunk_content)
    response = f'It says "{chunk_content}" (Source: [1]).'

    candidates = extract_citations(response, [chunk])
    assert len(candidates) == 1
    assert candidates[0].source_text == chunk_content


# ---------------------------------------------------------------------------
# M2-B1: rapidfuzz alignment fallback for quotes that aren't byte-for-byte
# substrings of the cited chunk (smart quotes, whitespace drift).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_alignment_fallback_finds_whitespace_drift_quote() -> None:
    """When byte-for-byte fails, partial_ratio_alignment locates the quote.

    The candidate's offsets point at the best-aligned span in the chunk;
    the verifier (Stage 2) re-checks at threshold 95 after normalizing.
    """

    chunk = _chunk(
        content="The   agreement\nshall terminate.",
        char_offset_start=100,
    )
    # Model emits the same text with normalized whitespace — byte-for-byte
    # find will miss; alignment fallback locates it.
    response = 'The model says "The agreement shall terminate." (Source: [1]).'

    candidates = extract_citations(response, [chunk])
    assert len(candidates) == 1
    cite = candidates[0]
    # The aligned span covers the whitespace-divergent region.
    assert cite.source_offset_start >= 100
    assert cite.source_offset_end <= 100 + len(chunk.content)
    # source_text is the model's quote verbatim; verifier normalizes both sides.
    assert cite.source_text == "The agreement shall terminate."


@pytest.mark.unit
def test_extract_alignment_fallback_rejects_unrelated_quote() -> None:
    """A quote with no real overlap with the chunk falls below threshold."""

    chunk = _chunk(content="The contract term is five years.")
    response = 'The model wrote "completely unrelated subject matter." (Source: [1]).'

    assert extract_citations(response, [chunk]) == []


@pytest.mark.unit
def test_extract_smart_quote_alignment_pairs() -> None:
    """Mixed shapes: a smart-quoted citation whose text differs in whitespace."""

    chunk = _chunk(content="The   agreement\nshall  terminate.")
    response = "The model says “The agreement shall terminate.” (Source: [1])."

    candidates = extract_citations(response, [chunk])
    assert len(candidates) == 1
    assert candidates[0].source_text == "The agreement shall terminate."


# ---------------------------------------------------------------------------
# DE-CIT-1: marker-less fallback. The model frequently quotes a source
# passage verbatim but omits the ``(Source: [N])`` tag entirely (observed
# in production: Spanish answers that quote a clause conversationally).
# When a quoted span is not tagged, locate it across ALL retrieved chunks
# (no index to trust) and, failing that, across the parent documents. The
# strict verifier still gates every candidate downstream.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_markerless_verbatim_quote_locates_across_chunks() -> None:
    """An untagged verbatim quote present in a retrieved chunk becomes a candidate."""

    chunk = _chunk(
        content="El plazo de confidencialidad será de cinco años.",
        char_offset_start=300,
    )
    # No (Source: [N]) tag — the model quoted the clause conversationally.
    response = 'El contrato dice: "El plazo de confidencialidad será de cinco años."'

    candidates = extract_citations(response, [chunk])

    assert len(candidates) == 1
    cite = candidates[0]
    assert cite.source_file_id == chunk.file_id
    assert cite.source_document_id == chunk.document_id
    assert cite.source_offset_start == 300 + chunk.content.find("El plazo")
    assert cite.source_text == "El plazo de confidencialidad será de cinco años."


@pytest.mark.unit
def test_markerless_locates_spanish_guillemets_quote() -> None:
    """Spanish «...» quotes without a tag are located too (corpus is es)."""

    chunk = _chunk(content="La obligación permanece vigente tras la terminación.")
    # The period sits outside the guillemets, so the quote excludes it; the
    # phrase is still an exact substring of the chunk.
    response = "El acuerdo establece que «La obligación permanece vigente tras la terminación»."

    candidates = extract_citations(response, [chunk])

    assert len(candidates) == 1
    assert candidates[0].source_text == "La obligación permanece vigente tras la terminación"


@pytest.mark.unit
def test_markerless_short_quote_is_ignored() -> None:
    """A short untagged quoted word must not become a spurious citation.

    Untagged quotes are inferred, not asserted by the model, so a single
    quoted word ("confidencial") that happens to appear byte-for-byte in a
    chunk would otherwise pass Stage 1 verification and render as a source.
    """

    chunk = _chunk(content="Toda la información confidencial del proyecto.")
    response = 'Se refiere a la información "confidencial" del proyecto.'

    assert extract_citations(response, [chunk]) == []


@pytest.mark.unit
def test_markerless_unfindable_quote_is_dropped() -> None:
    """A long untagged quote absent from every chunk is dropped."""

    chunk = _chunk(content="El contrato regula el plazo de entrega de la mercadería.")
    response = 'El modelo inventó "una cláusula que no está en ninguna parte del corpus."'

    assert extract_citations(response, [chunk]) == []


@pytest.mark.unit
def test_tagged_quote_is_not_double_counted_as_markerless() -> None:
    """A quote WITH a (Source: [N]) tag yields exactly one candidate."""

    chunk = _chunk(content="The contract term shall be five years.")
    response = 'The agreement says "The contract term shall be five years." (Source: [1]).'

    candidates = extract_citations(response, [chunk])

    assert len(candidates) == 1
    assert candidates[0].source_text == "The contract term shall be five years."


@pytest.mark.unit
def test_markerless_and_tagged_quotes_coexist() -> None:
    """One tagged and one untagged quote in the same answer → two candidates."""

    chunk1 = _chunk(content="First fact statement of some length.", char_offset_start=0)
    chunk2 = _chunk(
        content="Second fact assertion long enough to cite.",
        char_offset_start=500,
        page_start=3,
    )
    response = (
        'He said "First fact statement of some length." (Source: [1]) and '
        'also quoted "Second fact assertion long enough to cite." without a tag.'
    )

    candidates = extract_citations(response, [chunk1, chunk2])

    assert len(candidates) == 2
    by_doc = {c.source_document_id: c for c in candidates}
    assert by_doc[chunk1.document_id].source_text == "First fact statement of some length."
    assert by_doc[chunk2.document_id].source_text == "Second fact assertion long enough to cite."
    assert by_doc[chunk2.document_id].source_page == 3


@pytest.mark.unit
def test_markerless_full_document_fallback() -> None:
    """An untagged quote spanning a chunk boundary resolves via document scan."""

    doc_id = uuid.uuid4()
    file_id = uuid.uuid4()
    # The quote lives in the document but in neither chunk individually.
    chunk_a = _StubChunk(
        document_id=doc_id,
        file_id=file_id,
        content="...la primera parte de la cláusula relevante",
        page_start=1,
        char_offset_start=0,
        char_offset_end=44,
    )
    chunk_b = _StubChunk(
        document_id=doc_id,
        file_id=file_id,
        content="y la segunda parte que la completa...",
        page_start=2,
        char_offset_start=44,
        char_offset_end=81,
    )
    doc_content = (
        "la primera parte de la cláusula relevante y la segunda parte que la completa"
    )
    response = (
        'El contrato dice: "la primera parte de la cláusula relevante y la '
        'segunda parte que la completa".'
    )

    candidates = extract_citations(
        response, [chunk_a, chunk_b], {doc_id: doc_content}
    )

    assert len(candidates) == 1
    cite = candidates[0]
    assert cite.source_document_id == doc_id
    assert doc_content[cite.source_offset_start : cite.source_offset_end] == (
        "la primera parte de la cláusula relevante y la segunda parte que la completa"
    )


@pytest.mark.unit
def test_locate_in_chunk_public_exact_and_miss() -> None:
    """Test the public locate_in_chunk function directly."""

    from app.citation.extraction import locate_in_chunk

    content = "The Receiving Party shall hold Confidential Information in confidence."
    span = locate_in_chunk("hold Confidential Information", content)
    assert span is not None
    start, end = span
    assert content[start:end] == "hold Confidential Information"
    assert locate_in_chunk("text that is absent", content) is None
