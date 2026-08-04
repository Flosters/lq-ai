"""Citation extraction from assistant responses — quote-then-locate.

The model is instructed (via the RAG context-block system message) to
quote source passages in double quotes followed by ``(Source: [N])``
where ``N`` is the 1-based index of the retrieved chunk the quote came
from. This module parses that shape and resolves each quote to a
byte-precise span in the source document.

Resolution mechanics:

1. Regex over the response text matches both straight ``"..."`` and
   curly ``"..."`` quoted spans followed by ``(Source: [N])``.
2. For each match, look up ``retrieved_chunks[N - 1]``.
3. Locate the quote inside the chunk's content:

   * **Fast path:** byte-for-byte substring search via ``str.find``.
   * **Fallback (M2-B1):** when byte-for-byte misses, use
     :func:`rapidfuzz.fuzz.partial_ratio_alignment` to find the
     best-aligned span. Threshold 85 gates extraction — looser than
     the Stage 2 verifier's 95, so a candidate that survives
     extraction may still get rejected by verification. Quotes with
     no real overlap (an unrelated model fabrication) drop here.
   * **Chunk-boundary fallback (M3-0.2 / DE-277):** when the chunk-local
     search misses but the caller supplied ``document_contents``, retry
     the same exact-then-fuzzy locator against the full document text.
     This catches quotes that legitimately span two adjacent chunks
     (present in ``documents.normalized_content`` but in neither chunk
     individually) plus quotes where the model misnumbered the cited
     chunk index. The resolved offsets are document-absolute — no
     chunk-offset arithmetic is applied — so the downstream verifier
     (which already reads against ``document.normalized_content``)
     consumes them with no change.
4. Materialize a :class:`CitationCandidate` carrying the source
   document id + file id + byte-precise offsets + the model's quote
   verbatim. The verifier cascade (``app.citation.verification.verify``)
   then runs Stage 1 (byte-for-byte at offsets) and, on failure,
   Stage 2 (normalized fuzzy ratio).

Why a permissive extractor + strict verifier?
---------------------------------------------

Earlier M2-A2 extraction was strict (straight quotes + byte-for-byte
only) and dropped smart-quoted or whitespace-divergent citations
silently. With the M2-B1 Stage 2 verifier in place that policy hides
real wins — the verifier was built to forgive exactly those
normalization-only differences, but the candidates never reached it.
M2-B1 loosens extraction so the verifier sees more candidates and
ultimately the model's smart-quote citations land as
``verified=True, method='tolerant_match'``.

Chunk-mismatch observability (DE-277 option b)
----------------------------------------------

When the chunk-boundary fallback fires AND the chunk-local search had
missed, the extractor emits a structured ``citation_chunk_mismatch``
warning. The citation still verifies — quote-against-document is the
load-bearing correctness check — but the mismatch signal lets operators
spot model-side drift (e.g., the model consistently citing the wrong
``[N]`` index for a particular skill or prompt template).
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


# A protocol so the extractor accepts any chunk-shaped object — the
# production caller passes ``HybridSearchResult``; tests pass a stub.
class _RetrievedChunk(Protocol):
    document_id: uuid.UUID
    file_id: uuid.UUID
    content: str
    page_start: int | None
    char_offset_start: int
    char_offset_end: int


@dataclass(slots=True)
class CitationCandidate:
    """One extracted citation with byte-precise offsets, awaiting verification.

    All fields map 1-to-1 onto the ``message_citations`` row the
    persistence step will create. The verifier runs against this
    shape and stamps the ``verified`` / method / confidence flags.
    """

    source_file_id: uuid.UUID
    source_document_id: uuid.UUID
    source_offset_start: int
    source_offset_end: int
    source_page: int | None
    source_text: str


# Regex: matches both straight (``"..."``) and curly (``"..."``)
# double-quote pairs followed by ``(Source: [N])``. Alternation gives
# two named groups; one is non-None per match.
#
# ``re.DOTALL`` lets the body span line breaks. The body uses a lazy
# match (``+?``) so we don't gobble a closing quote that belongs to a
# later, different citation.
_CITATION_RE = re.compile(
    r'"(?P<sq>[^"]+?)"\s*\(Source:\s*\[(?P<sq_index>\d+)\]\)'
    r"|"
    r"“(?P<cq>[^”]+?)”\s*\(Source:\s*\[(?P<cq_index>\d+)\]\)",
    flags=re.DOTALL,
)

# DE-CIT-1: any double-quoted span — straight (``"..."``), curly
# (``"..."``), or Spanish guillemets (``«...»``) — with NO ``(Source: [N])``
# tag. The marker-less fallback scans these and locates them by content
# against the retrieved chunks. Guillemets matter: the corpus and the
# model's answers are Spanish, where ``«...»`` is the native quote shape.
_ANY_QUOTE_RE = re.compile(
    r'"(?P<sq>[^"]+?)"'
    r"|"
    r"“(?P<cq>[^”]+?)”"
    r"|"
    r"«(?P<gq>[^»]+?)»",
    flags=re.DOTALL,
)

# Minimum length (characters) for a marker-less quote to be considered.
# Tagged quotes are asserted by the model, so we trust short ones; an
# untagged quote is *inferred*, and a single quoted word (e.g. "razonable")
# that happens to appear verbatim in a chunk would pass Stage 1 verification
# and render as a spurious source. Requiring a phrase-length span keeps the
# fallback to genuine clause quotes without polluting the citation list.
_MARKERLESS_MIN_QUOTE_CHARS = 24

# Extraction-level fuzzy threshold. Below this, the quote is considered
# unrelated to the chunk content (model fabricated or mis-cited).
# Above, the candidate proceeds to the verifier cascade — Stage 1
# expects a byte-for-byte hit (rare after fallback fired) and Stage 2
# expects 95 on normalized text. Picking 85 here gives the Stage 2
# verifier a 10-point cushion to absorb meaningful normalization
# differences while filtering out obvious junk.
_ALIGNMENT_THRESHOLD = 85.0


def locate_in_chunk(quote: str, chunk_content: str) -> tuple[int, int] | None:
    """Find ``quote`` inside ``chunk_content`` — exact, then fuzzy.

    Returns the (start, end) offsets *into the chunk's content string*
    (not the document) when a credible match is found. ``None`` when
    no match survives the fuzzy threshold.
    """

    exact = chunk_content.find(quote)
    if exact >= 0:
        return exact, exact + len(quote)

    if not chunk_content or not quote:
        return None

    alignment = fuzz.partial_ratio_alignment(quote, chunk_content)
    if alignment is None or alignment.score < _ALIGNMENT_THRESHOLD:
        return None

    return alignment.dest_start, alignment.dest_end


def _candidate_at(
    chunk: _RetrievedChunk, start: int, end: int, quote: str
) -> CitationCandidate:
    """Build a candidate at the given (already document-absolute) offsets."""

    return CitationCandidate(
        source_file_id=chunk.file_id,
        source_document_id=chunk.document_id,
        source_offset_start=start,
        source_offset_end=end,
        source_page=chunk.page_start,
        source_text=quote,
    )


def _resolve_markerless_quote(
    quote: str,
    retrieved_chunks: Sequence[_RetrievedChunk],
    document_contents: Mapping[uuid.UUID, str] | None,
) -> CitationCandidate | None:
    """Locate an untagged quote across the retrieved chunks / their documents.

    There is no ``[N]`` index to trust, so we search everywhere, preferring
    exact matches over fuzzy ones globally: an exact hit in a parent
    document (a quote that spans two adjacent chunks) must win over a
    partial fuzzy hit inside one chunk. Order: exact-in-chunk →
    exact-in-document → fuzzy-in-chunk → fuzzy-in-document. The strict
    verifier re-checks whatever we return.
    """

    # One representative chunk per parent document — carries the file_id
    # and page we stamp on document-scan candidates.
    doc_reps: dict[uuid.UUID, _RetrievedChunk] = {}
    for chunk in retrieved_chunks:
        doc_reps.setdefault(chunk.document_id, chunk)

    # 1. Exact substring inside a chunk (chunk-local offsets).
    for chunk in retrieved_chunks:
        idx = chunk.content.find(quote)
        if idx >= 0:
            start = chunk.char_offset_start + idx
            return _candidate_at(chunk, start, start + len(quote), quote)

    # 2. Exact substring inside a parent document (document-absolute offsets).
    if document_contents is not None:
        for doc_id, chunk in doc_reps.items():
            doc_content = document_contents.get(doc_id)
            if not doc_content:
                continue
            idx = doc_content.find(quote)
            if idx >= 0:
                _log_markerless_document_scan(doc_id, quote)
                return _candidate_at(chunk, idx, idx + len(quote), quote)

    # 3. Fuzzy alignment inside a chunk.
    for chunk in retrieved_chunks:
        located = locate_in_chunk(quote, chunk.content)
        if located is not None:
            in_start, in_end = located
            return _candidate_at(
                chunk,
                chunk.char_offset_start + in_start,
                chunk.char_offset_start + in_end,
                quote,
            )

    # 4. Fuzzy alignment inside a parent document.
    if document_contents is not None:
        for doc_id, chunk in doc_reps.items():
            doc_content = document_contents.get(doc_id)
            if not doc_content:
                continue
            located = locate_in_chunk(quote, doc_content)
            if located is not None:
                _log_markerless_document_scan(doc_id, quote)
                return _candidate_at(chunk, located[0], located[1], quote)

    return None


def _log_markerless_document_scan(document_id: uuid.UUID, quote: str) -> None:
    logger.warning(
        "marker-less citation located via full-document scan",
        extra={
            "event": "citation_markerless_document_scan",
            "document_id": str(document_id),
            "quote_prefix": quote[:40],
        },
    )


def extract_citations(
    response_text: str,
    retrieved_chunks: Sequence[_RetrievedChunk],
    document_contents: Mapping[uuid.UUID, str] | None = None,
) -> list[CitationCandidate]:
    """Extract citations from the assistant response.

    Args:
        response_text: The full assistant message content.
        retrieved_chunks: The RAG-retrieved chunks delivered to the
            model in this turn's prompt, in the same order they were
            numbered ``[1], [2], …`` in the context block.
        document_contents: Optional map of document id →
            ``documents.normalized_content`` (M2-A1 surface). When
            supplied, enables the M3-0.2 / DE-277 chunk-boundary
            fallback: a quote that misses inside the cited chunk is
            retried against the full document text. When ``None`` (the
            default), the function behaves exactly as M2-B1 shipped —
            chunk-local search only.

    Returns:
        One :class:`CitationCandidate` per ``"..." (Source: [N])``
        pair (straight or curly quotes) whose quote could be located
        inside its cited chunk OR (when ``document_contents`` is
        supplied) inside the cited chunk's parent document. Pairs
        whose quote survives neither search, and pairs whose index
        is out of range, are dropped silently.
    """

    candidates: list[CitationCandidate] = []
    # Character spans of the tagged matches, so the marker-less pass below
    # doesn't re-process a quote the tagged pass already claimed.
    tagged_spans: list[tuple[int, int]] = []

    for match in _CITATION_RE.finditer(response_text):
        tagged_spans.append(match.span())
        # Exactly one of the two named branches is populated per match.
        quote = match.group("sq") or match.group("cq")
        index_str = match.group("sq_index") or match.group("cq_index")
        if quote is None or index_str is None:
            continue

        index_1based = int(index_str)
        if not (1 <= index_1based <= len(retrieved_chunks)):
            continue

        chunk = retrieved_chunks[index_1based - 1]
        located = locate_in_chunk(quote, chunk.content)
        if located is not None:
            in_chunk_start, in_chunk_end = located
            offset_start = chunk.char_offset_start + in_chunk_start
            offset_end = chunk.char_offset_start + in_chunk_end
        else:
            # M3-0.2 / DE-277: chunk-local search missed. Fall back to a
            # full-document scan when the caller supplied
            # normalized-content for the chunk's parent document. The
            # resolved offsets are document-absolute already (the
            # full document is the substring search space) so no
            # ``chunk.char_offset_start`` arithmetic is applied.
            doc_content = (
                document_contents.get(chunk.document_id) if document_contents is not None else None
            )
            if doc_content is None:
                # No full-document text available — either the caller
                # didn't supply the map, or the document was not loaded
                # (e.g., deleted between retrieval and persistence).
                # Drop the candidate; the model either fabricated the
                # quote or mis-cited an index pointing at content not
                # in the retrieved-chunk window.
                continue

            doc_located = locate_in_chunk(quote, doc_content)
            if doc_located is None:
                # Quote is not in the document at all. Drop.
                continue

            offset_start, offset_end = doc_located
            # DE-277 option (b): observability for chunk-mismatches.
            # The citation still verifies (quote-against-document is the
            # correctness check); the warning surfaces a model-side
            # signal worth investigating in aggregate.
            logger.warning(
                "citation chunk-boundary mismatch — quote located via "
                "full-document scan rather than cited chunk",
                extra={
                    "event": "citation_chunk_mismatch",
                    "document_id": str(chunk.document_id),
                    "cited_chunk_index": index_1based,
                    "quote_prefix": quote[:40],
                },
            )

        candidates.append(
            CitationCandidate(
                source_file_id=chunk.file_id,
                source_document_id=chunk.document_id,
                source_offset_start=offset_start,
                source_offset_end=offset_end,
                source_page=chunk.page_start,
                source_text=quote,
            )
        )

    # DE-CIT-1: marker-less fallback. Scan every quoted span that the
    # tagged pass did not claim; when it's a phrase-length quote we can
    # locate in the retrieved context, emit a candidate for the verifier.
    for match in _ANY_QUOTE_RE.finditer(response_text):
        q_start = match.start()
        if any(ts <= q_start < te for ts, te in tagged_spans):
            continue  # already handled as a tagged citation
        quote = match.group("sq") or match.group("cq") or match.group("gq")
        if quote is None or len(quote.strip()) < _MARKERLESS_MIN_QUOTE_CHARS:
            continue
        candidate = _resolve_markerless_quote(quote, retrieved_chunks, document_contents)
        if candidate is not None:
            candidates.append(candidate)

    return candidates
