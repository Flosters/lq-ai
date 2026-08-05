"""Executor tests for the M3-A2 LangGraph workflow.

Two layers:

* **Pure-helper tests** for the JSON coercion + summary aggregation
  in :mod:`app.playbooks.nodes` — no DB, no gateway.
* **End-to-end executor tests** that wire a real DB session (per the
  conftest fixture) + a stubbed gateway client. The stub returns
  hand-crafted ``ChatCompletionResponse`` payloads so the classify +
  redline nodes exercise their parse paths against deterministic JSON.

The stubbed gateway pattern intentionally avoids ``unittest.mock``
machinery for the response shape — the protocol surface is small and
returning a real ``SimpleNamespace`` keeps the mypy-style attribute
access in :func:`app.playbooks.nodes._dispatch_structured_call`
exercised honestly.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk
from app.models.file import File as FileModel
from app.models.playbook import Playbook, PlaybookExecution, PlaybookPosition
from app.models.user import User
from app.playbooks.executor import run_playbook_execution
from app.playbooks.nodes import (
    _coerce_chunk_indices,
    _coerce_confidence,
    _coerce_verdict,
    _parse_json_object,
    _shape_results_payload,
    _summarize,
    make_retrieve_node,
)
from app.security import hash_password

# ---------------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_coerce_verdict_accepts_canonical_values() -> None:
    for v in ("matches_standard", "matches_fallback", "deviates", "missing"):
        assert _coerce_verdict(v) == v


@pytest.mark.unit
def test_coerce_verdict_defaults_to_missing_for_unknown() -> None:
    assert _coerce_verdict("yes") == "missing"
    assert _coerce_verdict(None) == "missing"
    assert _coerce_verdict(42) == "missing"


@pytest.mark.unit
def test_coerce_confidence_normalizes_unknowns_to_low() -> None:
    assert _coerce_confidence("high") == "high"
    assert _coerce_confidence("medium") == "medium"
    assert _coerce_confidence("low") == "low"
    assert _coerce_confidence("very high") == "low"
    assert _coerce_confidence(None) == "low"


@pytest.mark.unit
def test_coerce_chunk_indices_filters_out_of_range() -> None:
    assert _coerce_chunk_indices([0, 1, 5, -1, "x", None], n_chunks=3) == [0, 1]
    assert _coerce_chunk_indices(None, n_chunks=3) == []
    assert _coerce_chunk_indices([], n_chunks=3) == []


@pytest.mark.unit
def test_parse_json_object_strips_code_fence() -> None:
    fenced = '```json\n{"verdict": "deviates", "confidence": "high"}\n```'
    parsed = _parse_json_object(fenced)
    assert parsed == {"verdict": "deviates", "confidence": "high"}


@pytest.mark.unit
def test_parse_json_object_returns_none_on_garbage() -> None:
    # None (not {}) signals an unusable response so the caller can tell a
    # parse FAILURE apart from a genuine "clause absent" verdict — a `{}`
    # would collapse both into the same confident-looking `missing`.
    assert _parse_json_object("not json at all") is None
    assert _parse_json_object("[1, 2, 3]") is None  # non-object
    assert _parse_json_object("") is None


@pytest.mark.unit
def test_parse_json_object_returns_none_on_truncated_string() -> None:
    # The Elementa failure mode: a long verbatim `matched_text` quote runs
    # past max_tokens and the JSON is cut off mid-string. json.loads raises
    # "Unterminated string"; the parser must surface that as None, never {}.
    truncated = (
        '{"verdict": "matches_standard",\n'
        ' "confidence": "high",\n'
        ' "matched_text": "The Receiving Party shall hold all Confidential '
        "Information disclosed by the Disclosing Party in strict confiden"
    )
    assert _parse_json_object(truncated) is None


@pytest.mark.unit
def test_summarize_counts_each_verdict_bucket() -> None:
    counts = _summarize(
        [
            {"verdict": "matches_standard"},
            {"verdict": "matches_standard"},
            {"verdict": "deviates"},
            {"verdict": "missing"},
            {"verdict": "matches_fallback"},
            {"verdict": "not_a_real_verdict"},  # ignored
        ]
    )
    assert counts == {
        "matches_standard": 2,
        "matches_fallback": 1,
        "deviates": 1,
        "missing": 1,
        "error": 0,
    }


@pytest.mark.unit
def test_shape_results_payload_includes_schema_version() -> None:
    state: dict[str, Any] = {
        "per_position_results": [{"verdict": "matches_standard"}],
    }
    payload = _shape_results_payload(state)  # type: ignore[arg-type]
    assert payload["schema_version"] == "m3-a2-v1"
    assert payload["summary"]["matches_standard"] == 1
    assert payload["positions"] == [{"verdict": "matches_standard"}]


# ---------------------------------------------------------------------------
# Stub gateway for executor end-to-end tests
# ---------------------------------------------------------------------------


@dataclass
class _StubMessage:
    content: str


@dataclass
class _StubChoice:
    message: _StubMessage


@dataclass
class _StubResponse:
    choices: list[_StubChoice]


@dataclass
class _StubGateway:
    """Returns a queued sequence of JSON-string responses, one per call.

    The executor calls :meth:`chat_completion` once per position for the
    classify node and once per deviating-position for the redline node.
    Tests pre-populate ``payloads`` with the JSON the LLM would have
    produced; the stub wraps each in the ``choices[0].message.content``
    shape :func:`_dispatch_structured_call` reads.
    """

    payloads: list[dict[str, Any]] = field(default_factory=list)
    calls_received: list[Any] = field(default_factory=list)

    async def chat_completion(self, request: Any) -> _StubResponse:
        self.calls_received.append(request)
        if not self.payloads:
            return _StubResponse(choices=[_StubChoice(message=_StubMessage(content=""))])
        payload = self.payloads.pop(0)
        return _StubResponse(
            choices=[_StubChoice(message=_StubMessage(content=json.dumps(payload)))]
        )


async def _make_user(db: AsyncSession) -> User:
    user = User(
        email=f"u-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password=hash_password("pw"),
        is_admin=False,
        role="member",
        mfa_enabled=False,
        must_change_password=False,
    )
    db.add(user)
    await db.flush()
    return user


async def _make_doc_with_chunks(
    db: AsyncSession,
    *,
    owner: User,
    normalized_text: str,
    chunks_text: list[str],
) -> tuple[FileModel, Document, list[DocumentChunk]]:
    f = FileModel(
        owner_id=owner.id,
        filename=f"doc-{uuid.uuid4().hex[:6]}.pdf",
        mime_type="application/pdf",
        size_bytes=2048,
        hash_sha256="d" * 64,
        storage_path=f"playbook-exec-fixture/{uuid.uuid4()}",
        ingestion_status="ready",
    )
    db.add(f)
    await db.flush()
    doc = Document(
        file_id=f.id,
        parser="pymupdf-only",
        parser_version="pymupdf=1.27",
        page_count=1,
        character_count=len(normalized_text),
        normalized_content=normalized_text,
        was_ocrd=False,
    )
    db.add(doc)
    await db.flush()
    chunks: list[DocumentChunk] = []
    offset = 0
    for i, text_value in enumerate(chunks_text):
        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=i,
            content=text_value,
            page_start=1,
            page_end=1,
            char_offset_start=offset,
            char_offset_end=offset + len(text_value),
        )
        db.add(chunk)
        chunks.append(chunk)
        offset += len(text_value)
    await db.flush()
    return f, doc, chunks


async def _make_playbook_with_position(
    db: AsyncSession,
    *,
    issue: str,
    detection_keywords: list[str],
    severity: str = "high",
    redline_strategy: str = "Tighten the clause to the standard.",
) -> Playbook:
    pb = Playbook(name="Test Playbook", contract_type="NDA")
    db.add(pb)
    await db.flush()
    pos = PlaybookPosition(
        playbook_id=pb.id,
        issue=issue,
        standard_language="The Receiving Party shall hold Confidential Information in confidence.",
        severity_if_missing=severity,
        detection_keywords=detection_keywords,
        detection_examples=[],
        redline_strategy=redline_strategy,
        fallback_tiers=[],
        position_order=0,
    )
    db.add(pos)
    await db.flush()
    return pb


# ---------------------------------------------------------------------------
# End-to-end executor tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_executor_classifies_matches_standard_end_to_end(
    db_session: AsyncSession,
) -> None:
    """Happy-path: one position, one matching chunk → completed + matches_standard."""
    owner = await _make_user(db_session)
    _file, doc, _chunks = await _make_doc_with_chunks(
        db_session,
        owner=owner,
        normalized_text=(
            "Section 1. The Receiving Party shall hold Confidential Information "
            "in confidence and not disclose it to any third party."
        ),
        chunks_text=[
            "Section 1. The Receiving Party shall hold Confidential Information",
            "in confidence and not disclose it to any third party.",
        ],
    )
    playbook = await _make_playbook_with_position(
        db_session,
        issue="Confidentiality",
        detection_keywords=["confidence", "Confidential"],
    )

    execution = PlaybookExecution(
        playbook_id=playbook.id,
        target_document_id=doc.id,
        user_id=owner.id,
    )
    db_session.add(execution)
    await db_session.flush()

    gateway = _StubGateway(
        payloads=[
            {
                "verdict": "matches_standard",
                "confidence": "high",
                "matched_fallback_rank": None,
                "matched_text": (
                    "The Receiving Party shall hold Confidential Information in confidence"
                ),
                "cited_chunk_indices": [0],
                "justification": "The clause materially matches the standard.",
            }
        ]
    )

    await run_playbook_execution(
        db_session,
        execution_id=execution.id,
        gateway=gateway,  # type: ignore[arg-type]
        judge_model="smart",
    )

    await db_session.refresh(execution)
    assert execution.status == "completed"
    assert execution.error is None
    assert execution.completed_at is not None
    assert execution.results is not None
    assert execution.results["schema_version"] == "m3-a2-v1"
    assert execution.results["summary"]["matches_standard"] == 1
    assert execution.results["summary"]["deviates"] == 0

    positions = execution.results["positions"]
    assert len(positions) == 1
    assert positions[0]["verdict"] == "matches_standard"
    assert positions[0]["redline"] is None
    # Classify cited chunk 0; verify the chunk_id surfaced.
    assert len(positions[0]["cited_chunk_ids"]) == 1


@pytest.mark.integration
async def test_executor_drafts_redline_for_deviates_verdict(
    db_session: AsyncSession,
) -> None:
    """A 'deviates' classification triggers a second LLM call for the redline."""
    owner = await _make_user(db_session)
    _file, doc, _chunks = await _make_doc_with_chunks(
        db_session,
        owner=owner,
        normalized_text=(
            "Section 2. Receiving Party may share Confidential Information with "
            "any affiliate or vendor at its sole discretion."
        ),
        chunks_text=[
            "Section 2. Receiving Party may share Confidential Information with",
            "any affiliate or vendor at its sole discretion.",
        ],
    )
    playbook = await _make_playbook_with_position(
        db_session,
        issue="Confidentiality",
        detection_keywords=["Confidential", "share"],
    )

    execution = PlaybookExecution(
        playbook_id=playbook.id,
        target_document_id=doc.id,
        user_id=owner.id,
    )
    db_session.add(execution)
    await db_session.flush()

    gateway = _StubGateway(
        payloads=[
            # 1st call — classify
            {
                "verdict": "deviates",
                "confidence": "high",
                "matched_fallback_rank": None,
                "matched_text": "Receiving Party may share Confidential Information with any affiliate or vendor at its sole discretion.",
                "cited_chunk_indices": [0, 1],
                "justification": "Permissive sharing is broader than the standard.",
            },
            # 2nd call — redline
            {
                "old_text": "may share Confidential Information with any affiliate or vendor at its sole discretion",
                "new_text": "shall hold Confidential Information in confidence and not disclose it without prior written consent",
                "justification": "Restores the confidentiality obligation per the standard.",
            },
        ]
    )

    await run_playbook_execution(
        db_session,
        execution_id=execution.id,
        gateway=gateway,  # type: ignore[arg-type]
    )

    await db_session.refresh(execution)
    assert execution.status == "completed"
    positions = execution.results["positions"]
    assert positions[0]["verdict"] == "deviates"
    assert positions[0]["redline"] is not None
    assert positions[0]["redline"]["old_text"].startswith("may share")
    assert "shall hold" in positions[0]["redline"]["new_text"]
    # Two gateway calls: classify + redline.
    assert len(gateway.calls_received) == 2


@pytest.mark.integration
async def test_executor_marks_missing_when_keyword_not_in_document(
    db_session: AsyncSession,
) -> None:
    """A position whose keywords don't match returns 'missing'; no redline call."""
    owner = await _make_user(db_session)
    _file, doc, _chunks = await _make_doc_with_chunks(
        db_session,
        owner=owner,
        normalized_text="Section 1. Generic boilerplate without any confidentiality language.",
        chunks_text=["Section 1. Generic boilerplate without any confidentiality language."],
    )
    playbook = await _make_playbook_with_position(
        db_session,
        issue="Limitation of Liability",
        detection_keywords=["liability", "indemnification"],
    )

    execution = PlaybookExecution(
        playbook_id=playbook.id,
        target_document_id=doc.id,
        user_id=owner.id,
    )
    db_session.add(execution)
    await db_session.flush()

    gateway = _StubGateway(
        payloads=[
            {
                "verdict": "missing",
                "confidence": "high",
                "matched_fallback_rank": None,
                "matched_text": "",
                "cited_chunk_indices": [],
                "justification": "No liability clause present in the contract.",
            }
        ]
    )

    await run_playbook_execution(
        db_session,
        execution_id=execution.id,
        gateway=gateway,  # type: ignore[arg-type]
    )

    await db_session.refresh(execution)
    assert execution.status == "completed"
    positions = execution.results["positions"]
    assert positions[0]["verdict"] == "missing"
    assert positions[0]["redline"] is None
    # Single gateway call: classify only — no redline for `missing`.
    assert len(gateway.calls_received) == 1


@pytest.mark.integration
async def test_executor_persists_error_verdict_on_gateway_failure(
    db_session: AsyncSession,
) -> None:
    """A gateway exception inside the classifier surfaces as an ``error`` verdict.

    The structured-call dispatcher swallows transport errors, but the
    executor must NOT pretend the clause is absent: a call that never
    produced a usable answer is indistinguishable, evidence-wise, from
    truncation, and mapping it to ``missing`` is the same dangerous
    false negative. The position surfaces as ``verdict='error'`` at low
    confidence. The execution still completes (status='completed')
    because the failure is per-position, not per-execution.
    """
    owner = await _make_user(db_session)
    _file, doc, _chunks = await _make_doc_with_chunks(
        db_session,
        owner=owner,
        normalized_text="Some content.",
        chunks_text=["Some content."],
    )
    playbook = await _make_playbook_with_position(
        db_session,
        issue="Test",
        detection_keywords=["content"],
    )

    execution = PlaybookExecution(
        playbook_id=playbook.id,
        target_document_id=doc.id,
        user_id=owner.id,
    )
    db_session.add(execution)
    await db_session.flush()

    class _FailingGateway:
        async def chat_completion(self, request: Any) -> Any:
            raise RuntimeError("gateway unreachable")

    await run_playbook_execution(
        db_session,
        execution_id=execution.id,
        gateway=_FailingGateway(),  # type: ignore[arg-type]
    )

    await db_session.refresh(execution)
    # The gateway swallowed the error per-call; the executor writes a
    # completed row but flags the position as 'error' — NOT 'missing' —
    # so a reviewer never mistakes an unreachable classifier for a
    # genuinely-absent clause.
    assert execution.status == "completed"
    positions = execution.results["positions"]
    assert positions[0]["verdict"] == "error"
    assert positions[0]["verdict"] != "missing"
    assert positions[0]["confidence"] == 0.5  # low → 0.5
    assert positions[0]["justification"]  # non-empty: explains the failure


# ---------------------------------------------------------------------------
# Retrieve node — multi-keyword FTS regression (the Elementa false-missing bug)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_retrieve_matches_any_keyword_not_all(
    db_session: AsyncSession,
) -> None:
    """A multi-keyword position must retrieve a chunk that hits ANY keyword.

    Regression for the Elementa NDA run where every position came back
    ``missing``: the retrieve node joined ``detection_keywords`` with
    spaces and fed them to ``websearch_to_tsquery``, which ANDs the
    terms. No single chunk held every keyword, so FTS returned nothing
    and the node fell back to the document's first chunks — the
    classifier then never saw the real clause and reported it missing.

    Here the governing-law clause lives only in the LAST chunk (index 5,
    outside the ``RETRIEVAL_TOP_K`` first-chunk fallback window) and only
    ONE of the position's keywords ("arbitration") appears in the whole
    document. Correct OR semantics must still surface that chunk.
    """
    owner = await _make_user(db_session)
    # Six chunks; only chunk 5 mentions arbitration. The first-chunks
    # fallback returns 0..3, so a chunk at index 5 proves real retrieval.
    chunks_text = [
        "Preamble. This Agreement is entered into between the parties.",
        "Definitions. Affiliate, Confidential Information, and Representatives.",
        "The Receiving Party shall hold Confidential Information in confidence.",
        "Permitted disclosures to Representatives on a need-to-know basis.",
        "Miscellaneous. Entire agreement, severability, and counterparts.",
        "Any dispute shall be finally settled by arbitration seated in Geneva.",
    ]
    _file, doc, _chunks = await _make_doc_with_chunks(
        db_session,
        owner=owner,
        normalized_text=" ".join(chunks_text),
        chunks_text=chunks_text,
    )

    position = {
        "id": str(uuid.uuid4()),
        "issue": "Governing Law and Venue",
        "detection_keywords": [
            "governing law",
            "jurisdiction",
            "venue",
            "arbitration",
        ],
    }
    state: dict[str, Any] = {
        "target_document_id": str(doc.id),
        "positions": [position],
    }

    retrieve = make_retrieve_node(db_session)
    result = await retrieve(state)  # type: ignore[arg-type]

    retrieved = {r["position_id"]: r["chunks"] for r in result["retrievals"]}
    chunks = retrieved[position["id"]]
    contents = " ".join(c["content"] for c in chunks)
    # The clause that matches one keyword must be retrieved. Under the
    # AND bug this fails: FTS returns nothing and the fallback hands back
    # only chunks 0..3, none of which mention arbitration.
    assert "arbitration" in contents.lower()


# ---------------------------------------------------------------------------
# Classify node — truncated JSON must NOT masquerade as "missing"
# (the second Elementa false-negative: a confident-looking absent clause)
# ---------------------------------------------------------------------------


@dataclass
class _TruncatedChoice:
    """A choice whose content is a truncated JSON string + finish_reason='length'.

    Models the provider cutting the response off at max_tokens mid-string —
    exactly what produced the ``Unterminated string`` log line on the
    Elementa run.
    """

    message: _StubMessage
    finish_reason: str = "length"


@dataclass
class _TruncatedGateway:
    """Returns one truncated, unparseable classify response, then empties."""

    content: str
    calls_received: list[Any] = field(default_factory=list)

    async def chat_completion(self, request: Any) -> Any:
        self.calls_received.append(request)
        return _StubResponse(  # type: ignore[return-value]
            choices=[_TruncatedChoice(message=_StubMessage(content=self.content))]
        )


@pytest.mark.integration
async def test_executor_truncated_classify_is_error_not_missing(
    db_session: AsyncSession,
) -> None:
    """A truncated classify response over a doc that DOES contain the clause
    must surface as ``error``, never a confident ``missing``.

    This is the core regression for the second Elementa bug: the standard
    confidentiality clause is present in the document (retrieval succeeds),
    but the model's classify JSON is cut off mid ``matched_text`` at
    max_tokens. The old code caught the ``JSONDecodeError``, returned
    ``{}``, and ``_coerce_verdict`` turned that into ``verdict='missing'``
    at confidence 0.5 with an empty justification — indistinguishable from
    a genuinely-absent clause, the worst failure mode in contract review.

    The fix: a parse failure is a distinct ``error`` verdict, so a reviewer
    never treats a truncated classifier answer as "the clause isn't there."
    """
    owner = await _make_user(db_session)
    _file, doc, _chunks = await _make_doc_with_chunks(
        db_session,
        owner=owner,
        normalized_text=(
            "Section 1. The Receiving Party shall hold Confidential Information "
            "in confidence and not disclose it to any third party."
        ),
        chunks_text=[
            "Section 1. The Receiving Party shall hold Confidential Information",
            "in confidence and not disclose it to any third party.",
        ],
    )
    playbook = await _make_playbook_with_position(
        db_session,
        issue="Confidentiality",
        detection_keywords=["confidence", "Confidential"],
    )

    execution = PlaybookExecution(
        playbook_id=playbook.id,
        target_document_id=doc.id,
        user_id=owner.id,
    )
    db_session.add(execution)
    await db_session.flush()

    # Valid JSON prefix, then cut off mid-string in matched_text — the
    # classic max_tokens truncation. json.loads raises "Unterminated string".
    truncated_content = (
        '{"verdict": "matches_standard",\n'
        ' "confidence": "high",\n'
        ' "matched_fallback_rank": null,\n'
        ' "cited_chunk_indices": [0],\n'
        ' "matched_text": "The Receiving Party shall hold Confidential '
        "Information in confidence and not disclose it to any third party, and "
        "shall protect such information using at least the same degree of care"
    )
    gateway = _TruncatedGateway(content=truncated_content)

    await run_playbook_execution(
        db_session,
        execution_id=execution.id,
        gateway=gateway,  # type: ignore[arg-type]
    )

    await db_session.refresh(execution)
    assert execution.status == "completed"
    positions = execution.results["positions"]
    assert len(positions) == 1
    # The whole point: a truncated answer is NOT a "clause absent" verdict.
    assert positions[0]["verdict"] == "error"
    assert positions[0]["verdict"] != "missing"
    # Low signal, non-empty justification explaining it's a classifier error.
    assert positions[0]["confidence"] == 0.5
    assert positions[0]["justification"]
    assert positions[0]["matched_text"] == ""
    assert positions[0]["redline"] is None
    # The error is counted in its own summary bucket, not folded into missing.
    assert execution.results["summary"]["error"] == 1
    assert execution.results["summary"]["missing"] == 0
