"""Integration tests for the LegVolution adapter endpoints.

The gateway client is mocked (same convention as test_enhance_prompt):
these tests exercise auth, request validation, prompt dispatch, JSON
parsing (fenced / bare / unparseable) and the {payload, request_id}
envelope — not a live provider.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.gateway import GatewayClient, get_gateway_client
from app.db.session import get_db
from app.main import app
from app.models import User
from app.schemas.gateway import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionResponse,
    ChatCompletionUsage,
)
from app.security import create_access_token, hash_password


def _override_get_db(db_session: AsyncSession):
    async def _override() -> AsyncIterator[AsyncSession]:
        yield db_session

    return _override


def _mock_gateway(model_text: str) -> AsyncMock:
    mock = AsyncMock(spec=GatewayClient)
    mock.chat_completion.return_value = ChatCompletionResponse(
        id="cmpl_test",
        created=0,
        model="claude-sonnet-4-6",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=model_text),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        routed_inference_tier=3,
        routed_provider="anthropic-prod",
    )
    return mock


@pytest_asyncio.fixture
async def caller(db_session: AsyncSession) -> User:
    user = User(
        email=f"legvolution-{uuid.uuid4().hex[:8]}@example.com",
        display_name="LegVolution Service",
        hashed_password=hash_password("correct-horse-battery-staple"),
        is_admin=False,
        mfa_enabled=False,
        must_change_password=False,
    )
    db_session.add(user)
    await db_session.flush()
    return user


def _bearer(user: User) -> dict[str, str]:
    token = create_access_token(user.id, user.email, is_admin=user.is_admin)
    return {"Authorization": f"Bearer {token}"}


def _client_with(*, db_session: AsyncSession, gateway_mock: AsyncMock) -> AsyncClient:
    app.dependency_overrides[get_db] = _override_get_db(db_session)
    app.dependency_overrides[get_gateway_client] = lambda: gateway_mock
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


def _cleanup() -> None:
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_gateway_client, None)


async def _post(
    db_session: AsyncSession,
    caller: User,
    gateway: AsyncMock,
    path: str,
    body: dict,
):
    try:
        async with _client_with(db_session=db_session, gateway_mock=gateway) as client:
            return await client.post(
                f"/api/v1/legvolution/{path}", headers=_bearer(caller), json=body
            )
    finally:
        _cleanup()


TERMS_JSON = """\
```json
[{"key": "contraparte", "label": "Contraparte", "value": "Proveedora del Sur"},
 {"key": "plazo", "label": "Plazo / vigencia", "value": "12 meses"}]
```
"""


@pytest.mark.integration
async def test_summarize_contract_returns_payload_and_request_id(
    db_session: AsyncSession, caller: User
) -> None:
    gateway = _mock_gateway(TERMS_JSON)
    resp = await _post(
        db_session,
        caller,
        gateway,
        "summarize-contract",
        {"text": "Contrato de servicios. Plazo 12 meses."},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["payload"][0]["key"] == "contraparte"
    assert uuid.UUID(body["request_id"])  # uuid válido
    # el request_id devuelto es el que se propagó al gateway
    _, kwargs = gateway.chat_completion.call_args
    assert kwargs["request_id"] == body["request_id"]
    # prompt en español, salida JSON exigida
    (gw_request,) = gateway.chat_completion.call_args.args
    assert "JSON" in gw_request.messages[0].content
    assert gw_request.messages[1].content.startswith("Contrato de servicios")


@pytest.mark.integration
async def test_extract_obligations_parses_bare_json(db_session: AsyncSession, caller: User) -> None:
    gateway = _mock_gateway(
        '[{"description": "Preaviso de 30 días", "due_date": null, "label": "Preaviso"}]'
    )
    resp = await _post(
        db_session, caller, gateway, "extract-obligations", {"text": "Preaviso de 30 días."}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["payload"][0]["label"] == "Preaviso"


@pytest.mark.integration
async def test_extract_appointments_altas_y_ceses(db_session: AsyncSession, caller: User) -> None:
    gateway = _mock_gateway(
        '[{"person_name": "Juan Pérez", "role": "director", "title": "Presidente",'
        ' "action": "alta", "since": "2026-03-05", "until": "2029-03-05"},'
        ' {"person_name": "Ana García", "role": "director", "title": "Directora Titular",'
        ' "action": "cese", "since": null, "until": "2026-03-05"}]'
    )
    resp = await _post(
        db_session,
        caller,
        gateway,
        "extract-appointments",
        {"text": "Acta de asamblea...", "kind": "acta_asamblea"},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()["payload"]
    assert {p["action"] for p in payload} == {"alta", "cese"}
    # la pista de tipo de acta viaja en el prompt de sistema
    (gw_request,) = gateway.chat_completion.call_args.args
    assert "acta_asamblea" in gw_request.messages[0].content


@pytest.mark.integration
async def test_check_acta_findings(db_session: AsyncSession, caller: User) -> None:
    gateway = _mock_gateway(
        '[{"key": "quorum", "severity": "warn", "label": "Quórum",'
        ' "detail": "No consta el capital presente"}]'
    )
    resp = await _post(
        db_session,
        caller,
        gateway,
        "check-acta",
        {"text": "Acta de asamblea sin quórum declarado", "kind": "acta_asamblea"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["payload"][0]["severity"] == "warn"


@pytest.mark.integration
async def test_run_workflow_dispatch_and_unknown(db_session: AsyncSession, caller: User) -> None:
    gateway = _mock_gateway(
        '{"summary": "Contrato razonable", "terms": [], "risks": [], "obligations": []}'
    )
    resp = await _post(
        db_session,
        caller,
        gateway,
        "run-workflow",
        {"workflow": "contract_review", "text": "Contrato…"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["payload"]["summary"] == "Contrato razonable"

    resp = await _post(
        db_session,
        caller,
        gateway,
        "run-workflow",
        {"workflow": "inexistente", "text": "x"},
    )
    assert resp.status_code == 422


@pytest.mark.integration
async def test_semantic_search_ranks_and_short_circuits_empty(
    db_session: AsyncSession, caller: User
) -> None:
    gateway = _mock_gateway('[{"id": "kb1", "score": 0.9, "snippet": "plazo de prescripción"}]')
    resp = await _post(
        db_session,
        caller,
        gateway,
        "semantic-search",
        {
            "query": "prescripción laboral",
            "documents": [
                {"id": "kb1", "text": "plazo de prescripción laboral 2 años"},
                {"id": "kb2", "text": "régimen de dividendos"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["payload"][0]["id"] == "kb1"

    # sin candidatos: no se llama al gateway
    gateway.chat_completion.reset_mock()
    resp = await _post(
        db_session,
        caller,
        gateway,
        "semantic-search",
        {"query": "prescripción laboral", "documents": []},
    )
    assert resp.status_code == 200
    assert resp.json()["payload"] == []
    gateway.chat_completion.assert_not_called()


@pytest.mark.integration
async def test_unparseable_model_output_is_502(db_session: AsyncSession, caller: User) -> None:
    gateway = _mock_gateway("Lo siento, no puedo analizar este documento.")
    resp = await _post(db_session, caller, gateway, "summarize-contract", {"text": "Contrato…"})
    assert resp.status_code == 502


@pytest.mark.integration
async def test_requires_auth(db_session: AsyncSession) -> None:
    gateway = _mock_gateway("[]")
    try:
        async with _client_with(db_session=db_session, gateway_mock=gateway) as client:
            resp = await client.post("/api/v1/legvolution/summarize-contract", json={"text": "x"})
    finally:
        _cleanup()
    assert resp.status_code == 401
