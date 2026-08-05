"""LangGraph nodes for the Playbook executor — M3-A2.

Four nodes run sequentially:

1. :func:`retrieve_node` — for each position, FTS over the target
   document's chunks using ``detection_keywords`` (lexical) to pick
   candidate clauses.
2. :func:`classify_node` — one structured-output LLM call per position
   producing ``matches_standard | matches_fallback | deviates | missing``
   plus a confidence and the chunks the verdict referenced.
3. :func:`redline_node` — for ``deviates`` verdicts only, a second
   structured-output LLM call drafts ``{old_text, new_text, justification}``
   per the position's ``redline_strategy``.
4. :func:`compile_node` — assembles the per-position results into the
   final ``playbook_executions.results`` JSONB payload and flips the
   execution row to ``completed`` (or ``error`` if a prior node set
   ``state["error"]``).

The dependencies the nodes need (DB session, gateway client, judge
model) live in node closures returned by the factories below — keeps
the node functions pure-ish over the state dict so LangGraph's merge
semantics stay clean.

Failure handling: any node may set ``state["error"]`` to short-circuit
later nodes. :func:`compile_node` checks for it and flips the execution
row to ``status='error'`` rather than ``'completed'``. Gateway / DB
exceptions inside a node bubble up; the executor catches them at the
graph-invocation boundary and updates the row similarly.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.gateway import GatewayClient
from app.models.document import DocumentChunk
from app.models.playbook import PlaybookExecution
from app.observability_helpers import get_tracer, record_attributes
from app.playbooks.state import (
    PlaybookExecutionState,
    PositionVerdict,
)
from app.schemas.gateway import ChatCompletionMessage, ChatCompletionRequest

logger = logging.getLogger(__name__)

# Top-k chunks retrieved per position. Keeps the classifier's context
# window bounded — the Playbook executor calls the LLM N times (once
# per position), so per-call cost matters. 4 chunks is enough to cover
# most clause spans while staying well under the typical 8K-input cap
# even on cheap models.
RETRIEVAL_TOP_K = 4

# Maximum tokens for the classify call. Raised from 600 after the
# Elementa NDA run truncated mid ``matched_text`` — the model quoted a
# long clause verbatim, blew the budget, and emitted an unterminated
# JSON string that parsed to nothing and was misreported as ``missing``.
#
# The ``smart`` alias resolves to a Claude Sonnet reasoning model, and
# ``max_tokens`` caps *thinking + visible output combined* (see the
# gateway's own note in ``gateway/app/config.py``). A reasoning-heavy
# call can therefore spend the whole budget thinking and emit only
# ``{"verdict": "`` before hitting ``finish_reason='length'`` — which is
# why 1500 still truncated the occasional position. 4000 leaves ample
# room for the model to think *and* finish the (now length-capped) JSON.
# Truncation should now be rare; when it still happens the executor
# surfaces it as an ``error`` verdict, never a false ``missing`` (see
# :func:`classify_node`). A cleaner long-term fix is to disable extended
# thinking for this deterministic extraction call at the gateway.
CLASSIFY_MAX_TOKENS = 4000

# Maximum tokens for the redline call. Raised to 4000 for the same
# reason as CLASSIFY_MAX_TOKENS: the ``smart`` judge is a reasoning
# model whose thinking tokens share this budget, and the redline output
# (verbatim old_text + new_text + justification) can itself be long. At
# 800 a redline for a long deviating clause truncated mid ``old_text``
# — benign (the position stays ``deviates`` with no auto-drafted edit,
# since a failed redline just yields empty fields) but it needlessly
# dropped suggestions the reviewer could have used.
REDLINE_MAX_TOKENS = 4000

# Classifier output JSON schema (documented in the prompt; parsed by
# :func:`_parse_classify_response`). The verdict + confidence pair
# mirrors the M2-C1 paraphrase-judge shape so future telemetry can
# unify on a single verdict-confidence schema across surfaces.
_VALID_VERDICTS: frozenset[str] = frozenset(
    {"matches_standard", "matches_fallback", "deviates", "missing"}
)
_VALID_CONFIDENCES: frozenset[str] = frozenset({"high", "medium", "low"})
_CONFIDENCE_NUMERIC: dict[str, float] = {"high": 0.9, "medium": 0.7, "low": 0.5}


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------


def make_retrieve_node(
    db: AsyncSession,
) -> Callable[[PlaybookExecutionState], Awaitable[dict[str, Any]]]:
    """Build the retrieve node bound to a DB session."""

    async def retrieve_node(state: PlaybookExecutionState) -> dict[str, Any]:
        target_doc_id = uuid.UUID(state["target_document_id"])
        retrievals: list[dict[str, Any]] = []

        for pos in state.get("positions", []):
            # Lexical FTS over the target document's chunks. The query
            # is the union of detection_keywords; we'd prefer to also
            # mix in detection_examples via vector search, but per-doc
            # embedding search adds a layer of complexity the executor
            # skeleton doesn't need to ship with — the keyword path
            # works well for the high-signal contract-clause case
            # (counterparty / cap / term / governing law), and the
            # M3-A2 spec accepts this scope.
            keywords = pos.get("detection_keywords") or []
            if not keywords:
                # No keywords supplied → take the first chunk as a
                # defensive fallback so the classifier still sees the
                # document. This is the "missing keyword" failure mode
                # operators see if they author a playbook position
                # without populating ``detection_keywords``.
                logger.info(
                    "playbook_executor.retrieve: position has no detection_keywords",
                    extra={
                        "event": "playbook_retrieve_no_keywords",
                        "position_id": pos["id"],
                        "issue": pos["issue"],
                    },
                )
                fallback = await _fetch_first_chunks(db, target_doc_id, limit=RETRIEVAL_TOP_K)
                retrievals.append({"position_id": pos["id"], "chunks": fallback})
                continue

            chunks = await _fts_over_document(
                db,
                document_id=target_doc_id,
                keywords=keywords,
                limit=RETRIEVAL_TOP_K,
            )
            if not chunks:
                # No FTS hits — fall back to the document's first
                # chunks so the classifier still has document context
                # to evaluate. The classifier's verdict in this case
                # will typically be ``missing`` (the position's clause
                # isn't in the doc).
                chunks = await _fetch_first_chunks(db, target_doc_id, limit=RETRIEVAL_TOP_K)
            retrievals.append({"position_id": pos["id"], "chunks": chunks})

        return {"retrievals": retrievals}

    return retrieve_node


async def _fts_over_document(
    db: AsyncSession,
    *,
    document_id: uuid.UUID,
    keywords: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Run FTS over ``document_chunks`` scoped to one document.

    A chunk matches if it hits **any** of the position's
    ``detection_keywords`` — the intent when an author lists several. We
    build that as an OR of one ``plainto_tsquery`` per keyword combined
    with ``||``: ``plainto_tsquery`` ANDs the words *inside* a phrase
    (so ``"governing law"`` stays a phrase), while ``||`` ORs *across*
    keywords.

    ``websearch_to_tsquery('english', " ".join(keywords))`` was used here
    before and is deliberately avoided: it treats spaces as AND, so a
    space-joined keyword string requires a single chunk to contain
    *every* keyword. Multi-keyword positions then matched nothing, fell
    back to the document's first chunks, and were misreported as
    ``missing`` — the false-negative this replaces.
    """
    terms = [k.strip() for k in keywords if k and k.strip()]
    if not terms:
        return []

    # Build ``plainto_tsquery(:k0) || plainto_tsquery(:k1) || ...``. Only
    # the fixed ``plainto_tsquery('english', :kN)`` fragments are
    # interpolated into the SQL; every keyword travels as a bound
    # parameter, so this is not an injection vector.
    params: dict[str, Any] = {"doc_id": str(document_id), "limit": limit}
    or_parts: list[str] = []
    for i, term in enumerate(terms):
        params[f"k{i}"] = term
        or_parts.append(f"plainto_tsquery('english', :k{i})")
    tsquery = " || ".join(or_parts)

    result = await db.execute(
        text(
            "SELECT dc.id::text, dc.chunk_index, dc.content, "
            "dc.char_offset_start, dc.char_offset_end, dc.page_start, "
            f"ts_rank_cd(dc.content_tsv, {tsquery}) AS rank "
            "FROM document_chunks dc "
            "WHERE dc.document_id = :doc_id "
            f"AND dc.content_tsv @@ ({tsquery}) "
            "ORDER BY rank DESC, dc.chunk_index ASC "
            "LIMIT :limit"
        ),
        params,
    )
    return [
        {
            "id": row.id,
            "chunk_index": row.chunk_index,
            "content": row.content,
            "char_offset_start": row.char_offset_start,
            "char_offset_end": row.char_offset_end,
            "page_start": row.page_start,
        }
        for row in result
    ]


async def _fetch_first_chunks(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Defensive fallback when the FTS yields no rows for a position."""
    stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        {
            "id": str(row.id),
            "chunk_index": row.chunk_index,
            "content": row.content,
            "char_offset_start": row.char_offset_start,
            "char_offset_end": row.char_offset_end,
            "page_start": row.page_start,
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------


_CLASSIFY_SYSTEM_PROMPT = """\
You are a Contract Review Classifier for a legal AI assistant.

You will be given:
* The org's STANDARD POSITION on a contract issue (the preferred clause).
* A ranked list of FALLBACK TIERS — acceptable alternatives to the standard.
* The actual CONTRACT EXCERPT to evaluate.

Your job: determine how the contract excerpt compares to the standard
+ fallbacks. Output STRICTLY VALID JSON in this exact shape:

  {"verdict": "matches_standard" | "matches_fallback" | "deviates" | "missing",
   "confidence": "high" | "medium" | "low",
   "matched_fallback_rank": <int|null>,
   "matched_text": "<short verbatim quote — the operative sentence or clause span your verdict references; at most ~60 words. Do NOT paste an entire long clause verbatim. Empty if missing.>",
   "cited_chunk_indices": [<int>, ...],
   "justification": "<one or two sentences explaining your verdict>"}

Verdict meanings:

* "matches_standard" — the contract's clause is materially equivalent
  to the standard position (same legal effect; wording may differ).
* "matches_fallback" — the contract's clause matches one of the
  fallback tiers. Populate ``matched_fallback_rank`` with the matched
  tier's rank (1 = preferred fallback; higher numbers = weaker).
* "deviates" — the contract addresses the issue but the clause is
  worse than every listed fallback. The redliner will draft a
  suggested edit using the position's redline_strategy.
* "missing" — the contract does NOT address the issue at all. Leave
  ``matched_text`` empty.

Confidence meanings:

* "high" — the verdict is unambiguous; another careful reader would
  reach the same conclusion.
* "medium" — the verdict is supported but reasonable readers might
  disagree.
* "low" — the source evidence is thin or ambiguous; flag for human
  review.

The ``cited_chunk_indices`` field is a list of the 0-based indices of
the chunks (in the order they were presented below) whose content
supports your verdict. Always include at least one index for
non-``missing`` verdicts.

Bias toward "low" confidence and toward ``missing`` on uncertainty.
False positives — calling a missing position "matches" — do more
damage in contract review than false negatives.
"""


def make_classify_node(
    *,
    gateway: GatewayClient,
    judge_model: str,
) -> Callable[[PlaybookExecutionState], Awaitable[dict[str, Any]]]:
    """Build the classify node bound to a gateway client + model alias."""

    async def classify_node(state: PlaybookExecutionState) -> dict[str, Any]:
        # Typed-as-Any because the chunk items are TypedDict
        # (``_ChunkForRetrieval``); mypy refuses to widen them to plain
        # dicts. Same posture as :func:`redline_node`.
        retrievals_by_position: dict[str, list[Any]] = {
            r["position_id"]: list(r["chunks"]) for r in state.get("retrievals", [])
        }
        results: list[dict[str, Any]] = []

        tracer = get_tracer()
        for pos in state.get("positions", []):
            with tracer.start_as_current_span("playbook.position") as pos_span:
                record_attributes(
                    pos_span,
                    **{
                        "playbook.position.id": str(pos["id"]),
                        "playbook.position.order": pos.get("position_order"),
                    },
                )
                chunks = retrievals_by_position.get(pos["id"], [])
                messages = _build_classify_messages(pos, chunks)
                call = await _dispatch_structured_call(
                    gateway=gateway,
                    model=judge_model,
                    messages=messages,
                    max_tokens=CLASSIFY_MAX_TOKENS,
                )

                if call.data is None:
                    # The classify call produced no usable answer (transport
                    # failure, or a truncated / malformed JSON body). Do NOT
                    # coerce this to a confident ``missing`` — that is
                    # indistinguishable from a genuinely-absent clause and is
                    # the exact false negative that shipped the Elementa run.
                    # Surface a distinct ``error`` verdict for human review.
                    record_attributes(
                        pos_span,
                        **{
                            "playbook.position.verdict": "error",
                            "playbook.classify.failure": call.failure or "unknown",
                            "playbook.classify.finish_reason": call.finish_reason or "",
                        },
                    )
                    logger.warning(
                        "playbook classify produced no usable answer; "
                        "emitting 'error' verdict (failure=%s, finish_reason=%s)",
                        call.failure,
                        call.finish_reason,
                        extra={
                            "event": "playbook_classify_error_verdict",
                            "position_id": pos["id"],
                            "failure": call.failure,
                            "finish_reason": call.finish_reason,
                        },
                    )
                    results.append(
                        _error_verdict_result(
                            pos,
                            failure=call.failure,
                            finish_reason=call.finish_reason,
                        )
                    )
                    continue

                verdict_data = call.data
                verdict = _coerce_verdict(verdict_data.get("verdict"))
                confidence_str = _coerce_confidence(verdict_data.get("confidence"))
                cited_indices = _coerce_chunk_indices(
                    verdict_data.get("cited_chunk_indices"), n_chunks=len(chunks)
                )
                cited_chunk_ids = [chunks[i]["id"] for i in cited_indices] if chunks else []
                matched_text = str(verdict_data.get("matched_text") or "")
                justification = str(verdict_data.get("justification") or "")
                matched_fallback_rank_raw = verdict_data.get("matched_fallback_rank")
                matched_fallback_rank: int | None
                if verdict == "matches_fallback" and isinstance(matched_fallback_rank_raw, int):
                    matched_fallback_rank = matched_fallback_rank_raw
                else:
                    matched_fallback_rank = None

                results.append(
                    {
                        "position_id": pos["id"],
                        "issue": pos["issue"],
                        "severity_if_missing": pos["severity_if_missing"],
                        "verdict": verdict,
                        "confidence": _CONFIDENCE_NUMERIC[confidence_str],
                        "matched_fallback_rank": matched_fallback_rank,
                        "cited_chunk_ids": cited_chunk_ids,
                        "matched_text": matched_text,
                        "redline": None,
                        "justification": justification,
                    }
                )

        return {"per_position_results": results}

    return classify_node


def _error_verdict_result(
    position: Any,
    *,
    failure: str | None,
    finish_reason: str | None,
) -> dict[str, Any]:
    """Build a per-position result for a classify call that produced no answer.

    Emitted when :func:`_dispatch_structured_call` returns ``data is
    None``. The verdict is ``error`` — never ``missing`` — so the outcome
    reads as "the classifier failed here," not "the clause is absent."
    Confidence is pinned ``low`` and the justification says in words that
    this is not an absence determination, so nothing downstream (UI, Word
    add-in, summary counts) can present it as a confident finding.
    """
    detail = failure or "unknown"
    if finish_reason:
        detail = f"{detail}, finish_reason={finish_reason}"
    reason = (
        "the model response was truncated before it finished"
        if finish_reason == "length"
        else "the model did not return a usable classification"
    )
    return {
        "position_id": position["id"],
        "issue": position["issue"],
        "severity_if_missing": position["severity_if_missing"],
        "verdict": "error",
        "confidence": _CONFIDENCE_NUMERIC["low"],
        "matched_fallback_rank": None,
        "cited_chunk_ids": [],
        "matched_text": "",
        "redline": None,
        "justification": (
            f"Classification could not be completed: {reason} ({detail}). "
            "This is a classifier error, NOT a determination that the clause "
            "is absent — this position needs human review."
        ),
    }


def _build_classify_messages(
    position: Any,
    chunks: list[Any],
) -> list[ChatCompletionMessage]:
    """Render the classify-prompt messages for one position + its retrieved chunks.

    Typed as :data:`Any` for the same TypedDict-widening reason as
    :func:`_build_redline_messages` (q.v.).
    """
    fallback_lines: list[str] = []
    for tier in position.get("fallback_tiers") or []:
        rank = tier.get("rank")
        description = tier.get("description") or ""
        language = tier.get("language") or ""
        fallback_lines.append(f"Tier {rank} — {description}\nLanguage:\n{language}")

    chunk_blocks: list[str] = []
    for i, chunk in enumerate(chunks):
        chunk_blocks.append(f"[CHUNK {i}]\n{chunk['content']}")

    user_content = (
        f"ISSUE: {position['issue']}\n\n"
        f"STANDARD POSITION:\n{position['standard_language']}\n\n"
        f"FALLBACK TIERS:\n"
        + ("\n\n".join(fallback_lines) if fallback_lines else "(none specified)")
        + "\n\n"
        + "CONTRACT EXCERPT:\n"
        + ("\n\n".join(chunk_blocks) if chunk_blocks else "(no chunks retrieved)")
    )
    return [
        ChatCompletionMessage(role="system", content=_CLASSIFY_SYSTEM_PROMPT),
        ChatCompletionMessage(role="user", content=user_content),
    ]


# ---------------------------------------------------------------------------
# Redline
# ---------------------------------------------------------------------------


_REDLINE_SYSTEM_PROMPT = """\
You are a Contract Redline Drafter for a legal AI assistant.

You will be given:
* The org's STANDARD POSITION on a contract issue.
* The contract's CURRENT CLAUSE that deviates from the standard.
* The REDLINE STRATEGY — instructions on how to redline this position.

Draft a tracked-changes redline. Output STRICTLY VALID JSON:

  {"old_text": "<verbatim text from the contract that should be replaced>",
   "new_text": "<suggested replacement text>",
   "justification": "<one or two sentences explaining why the change improves the contract>"}

The ``old_text`` field MUST be a verbatim substring of the contract
clause provided — the operator's editor will use it for textual
matching. If you cannot identify a single contiguous span to replace,
return ``old_text`` as the empty string and explain in
``justification``.

Preserve the contract's drafting style — match definite/indefinite
articles, capitalization, and tense. Don't introduce defined terms
that aren't already in the contract.
"""


def make_redline_node(
    *,
    gateway: GatewayClient,
    judge_model: str,
) -> Callable[[PlaybookExecutionState], Awaitable[dict[str, Any]]]:
    """Build the redline node bound to a gateway client + model alias.

    Iterates :func:`classify_node`'s ``per_position_results`` and runs a
    structured-output LLM call for each ``"deviates"`` row, populating
    its ``redline`` field. Non-deviating rows pass through unchanged.
    """

    positions_index: dict[str, dict[str, Any]] = {}

    async def redline_node(state: PlaybookExecutionState) -> dict[str, Any]:
        # Rebuild the per-position index for redline_strategy lookup.
        # Typed-as-Any because the position items are TypedDict; mypy
        # narrows them to ``_PositionInput`` and refuses the
        # ``dict[str, Any]`` value annotation otherwise.
        nonlocal positions_index
        positions_index = {pos["id"]: dict(pos) for pos in state.get("positions", [])}

        updated: list[Any] = []
        for result in state.get("per_position_results", []):
            if result.get("verdict") != "deviates":
                updated.append(result)
                continue

            pos = positions_index.get(result["position_id"])
            if pos is None:
                # Defensive: the position vanished between classify and
                # redline. Pass through without a redline.
                updated.append(result)
                continue

            messages = _build_redline_messages(pos, result)
            call = await _dispatch_structured_call(
                gateway=gateway,
                model=judge_model,
                messages=messages,
                max_tokens=REDLINE_MAX_TOKENS,
            )
            # Best-effort: a failed redline call just yields empty fields
            # (no suggested edit). Unlike classify, there's no false-negative
            # risk here — the deviation is already flagged; the redline is an
            # optional convenience the operator can still author by hand.
            redline_data = call.data or {}
            redline = {
                "old_text": str(redline_data.get("old_text") or ""),
                "new_text": str(redline_data.get("new_text") or ""),
                "justification": str(redline_data.get("justification") or ""),
            }
            updated.append({**result, "redline": redline})

        return {"per_position_results": updated}

    return redline_node


def _build_redline_messages(
    position: Any,
    classify_result: Any,
) -> list[ChatCompletionMessage]:
    """Render the redline-prompt messages for one deviating position.

    Typed as :data:`Any` because the runtime callers pass TypedDict
    instances (``_PositionInput`` / ``_PositionResult``) that
    structurally satisfy the dict access here. mypy doesn't widen
    TypedDicts to plain dicts automatically; the structural-only
    access keeps this safe.
    """
    user_content = (
        f"ISSUE: {position['issue']}\n\n"
        f"STANDARD POSITION:\n{position['standard_language']}\n\n"
        f"REDLINE STRATEGY:\n{position.get('redline_strategy') or '(none specified)'}\n\n"
        f"CONTRACT'S CURRENT CLAUSE:\n{classify_result.get('matched_text') or ''}"
    )
    return [
        ChatCompletionMessage(role="system", content=_REDLINE_SYSTEM_PROMPT),
        ChatCompletionMessage(role="user", content=user_content),
    ]


# ---------------------------------------------------------------------------
# Compile + persist
# ---------------------------------------------------------------------------


def make_compile_node(
    db: AsyncSession,
) -> Callable[[PlaybookExecutionState], Awaitable[dict[str, Any]]]:
    """Build the compile node bound to a DB session.

    The compile node writes the assembled results into
    ``playbook_executions`` and flips status to ``'completed'`` (or
    ``'error'`` if a prior node set ``state["error"]``). It also
    populates ``completed_at``.
    """

    async def compile_node(state: PlaybookExecutionState) -> dict[str, Any]:
        execution_id = uuid.UUID(state["execution_id"])
        results_payload = _shape_results_payload(state)
        error = state.get("error")

        if error:
            await db.execute(
                update(PlaybookExecution)
                .where(PlaybookExecution.id == execution_id)
                .values(
                    status="error",
                    error=error,
                    results=results_payload,
                    completed_at=datetime.now(UTC),
                )
            )
        else:
            await db.execute(
                update(PlaybookExecution)
                .where(PlaybookExecution.id == execution_id)
                .values(
                    status="completed",
                    results=results_payload,
                    completed_at=datetime.now(UTC),
                )
            )
        await db.commit()
        return {}

    return compile_node


def _shape_results_payload(state: PlaybookExecutionState) -> dict[str, Any]:
    """Render the per-position results into the JSONB payload shape."""
    per_position = state.get("per_position_results", [])
    summary = _summarize(per_position)
    return {
        "schema_version": "m3-a2-v1",
        "positions": per_position,
        "summary": summary,
    }


def _summarize(per_position: list[Any]) -> dict[str, int]:
    """Aggregate per-position verdict counts for the UI summary card.

    Typed as ``list[Any]`` because the runtime callers pass a list of
    ``_PositionResult`` TypedDicts; mypy doesn't widen TypedDicts to
    plain dicts. The aggregation only reads the ``verdict`` key.
    """
    counts = {
        "matches_standard": 0,
        "matches_fallback": 0,
        "deviates": 0,
        "missing": 0,
        "error": 0,
    }
    for result in per_position:
        verdict = result.get("verdict")
        if verdict in counts:
            counts[verdict] += 1
    return counts


# ---------------------------------------------------------------------------
# Structured-output dispatcher
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredCallResult:
    """Outcome of a structured-JSON LLM call.

    ``data`` is the parsed object on success and ``None`` on *any*
    failure — transport error, empty/malformed response envelope, or
    JSON that could not be parsed (typically a truncation). ``failure``
    carries a short machine code for the ``None`` case; ``finish_reason``
    is the provider's stop reason when the response body reached us
    (``"length"`` is the truncation signal).

    The ``None``-means-failure contract is the crux of the Elementa fix:
    callers can now tell a call that *failed* apart from one that
    *succeeded and said the clause is absent*. Best-effort callers (the
    redliner) keep their old behaviour via ``data or {}``; the classifier
    keys off ``data is None`` to emit a distinct ``error`` verdict rather
    than a confident-looking ``missing``.
    """

    data: dict[str, Any] | None
    failure: str | None = None
    finish_reason: str | None = None


async def _dispatch_structured_call(
    *,
    gateway: GatewayClient,
    model: str,
    messages: list[ChatCompletionMessage],
    max_tokens: int,
) -> StructuredCallResult:
    """Run a structured-JSON LLM call and return a :class:`StructuredCallResult`.

    Mirrors the M2-C1 paraphrase-judge dispatch pattern. ``temperature``
    is omitted: Anthropic Opus 4.x reasoning models rejected the
    parameter as of 2026-05, and the gateway only forwards non-None
    values to providers. Determinism for reasoning models is implicit;
    sampled models retain their provider default. ``anonymize=False``
    because the classifier needs to see the actual contract text to
    verify it; ``lq_ai_purpose='playbook_executor'`` so the routing log
    can be filtered for cost calibration.

    On failure ``data`` is ``None`` and ``failure`` names the stage that
    broke; the caller decides how to surface it (the classifier as an
    ``error`` verdict, the redliner as an empty redline).
    """
    request = ChatCompletionRequest(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        anonymize=False,
        lq_ai_purpose="playbook_executor",
    )
    try:
        response = await gateway.chat_completion(request)
    except Exception as exc:
        logger.warning(
            "playbook structured-call gateway error: %s",
            exc,
            extra={
                "event": "playbook_structured_call_error",
                "error_type": type(exc).__name__,
            },
        )
        return StructuredCallResult(data=None, failure="gateway_error")

    try:
        choices = response.choices
        if not choices:
            return StructuredCallResult(data=None, failure="no_choices")
        choice = choices[0]
        content = choice.message.content
    except AttributeError:
        return StructuredCallResult(data=None, failure="malformed_response")

    finish_reason = getattr(choice, "finish_reason", None)
    if not content:
        return StructuredCallResult(
            data=None, failure="empty_content", finish_reason=finish_reason
        )

    parsed = _parse_json_object(content, finish_reason=finish_reason)
    if parsed is None:
        return StructuredCallResult(
            data=None, failure="malformed_json", finish_reason=finish_reason
        )
    return StructuredCallResult(data=parsed, finish_reason=finish_reason)


def _parse_json_object(
    content: str, *, finish_reason: str | None = None
) -> dict[str, Any] | None:
    """Lenient JSON parse — trim a leading code fence if present, then ``json.loads``.

    Returns ``None`` (not ``{}``) on any failure so the caller can tell a
    parse FAILURE apart from a model that genuinely returned an object:
    a truncated response and a real ``missing`` verdict must never
    collapse into the same confident-looking outcome. On a decode error
    the log carries ``finish_reason`` and a prefix of the raw content so
    operators can confirm truncation (``finish_reason='length'``) as the
    cause rather than guessing.
    """
    stripped = content.strip()
    if stripped.startswith("```"):
        # Drop a leading ``` or ```json fence and the trailing ``` line.
        stripped = stripped.split("```", 2)[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.rstrip("`").strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        logger.warning(
            "playbook structured-call returned malformed JSON "
            "(finish_reason=%s, likely_truncated=%s): %s | raw_prefix=%r",
            finish_reason,
            finish_reason == "length",
            exc,
            content[:200],
            extra={
                "event": "playbook_structured_call_malformed_json",
                "finish_reason": finish_reason,
                "likely_truncated": finish_reason == "length",
                "raw_content_prefix": content[:500],
            },
        )
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _coerce_verdict(raw: Any) -> PositionVerdict:
    """Normalize the model's verdict to the canonical enum or default ``missing``."""
    if isinstance(raw, str) and raw in _VALID_VERDICTS:
        return raw  # type: ignore[return-value]
    return "missing"


def _coerce_confidence(raw: Any) -> str:
    """Normalize confidence to one of the three valid values; default ``low``."""
    if isinstance(raw, str) and raw in _VALID_CONFIDENCES:
        return raw
    return "low"


def _coerce_chunk_indices(raw: Any, *, n_chunks: int) -> list[int]:
    """Filter the model-emitted index list to valid 0-based ints inside the chunk count."""
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        if isinstance(item, int) and 0 <= item < n_chunks:
            out.append(item)
    return out
