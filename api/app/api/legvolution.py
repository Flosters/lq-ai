"""LegVolution adapter endpoints.

Thin JSON-in / JSON-out surface consumed by the LegVolution legal
platform's ``LQAIClient`` (backend/app/services/lqai.py in that repo).
Each endpoint wraps one prompt against the Inference Gateway and returns
``{"payload": <parsed JSON>, "request_id": <uuid>}``. The ``request_id``
is forwarded to the gateway as ``X-Request-Id`` so the caller can
correlate its audit_log rows with the gateway's inference_routing_log.

Design notes:

* Prompts are in Spanish (the platform's working language) and demand a
  single JSON value in the response. Parsing is tolerant (fenced block
  or bare JSON) but a response that cannot be parsed raises 502 — the
  LegVolution client maps any non-200 to its ``LQAIError`` and falls
  back to local heuristics, so an unparseable model output degrades
  gracefully instead of propagating garbage.
* Every executed inference writes one ``audit_log`` row — action
  ``legvolution.<endpoint>``, resource_type ``legvolution`` — carrying
  the ``request_id`` that also went to the gateway, the routing tier and
  provider it reported back, and ``on_behalf_of``. LegVolution
  authenticates as a single service account, so without that last field
  lq-ai could not tell *which person* originated an inference; the
  caller passes the real user's email and lq-ai records it. Calls with
  no acting user (automations, jobs) leave it null, which is the truth.
* Auth is the standard bearer token; LegVolution logs in with a service
  account via ``POST /api/v1/auth/login``.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import ActiveUser
from app.audit import audit_action
from app.clients.gateway import GatewayClient, get_gateway_client
from app.db.session import get_db
from app.schemas.gateway import ChatCompletionMessage, ChatCompletionRequest

log = logging.getLogger(__name__)

router = APIRouter(prefix="/legvolution", tags=["legvolution"])

# Analysis calls favor quality over latency but none of them need the
# top tier; the operator can remap the alias in gateway.yaml.
DEFAULT_MODEL_ALIAS = "fast"

_TEXT_MAX = 200_000  # ~50k tokens of contract/acta text; defensive cap

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)

_SYSTEM_PREAMBLE = (
    "Sos un asistente legal para una plataforma corporativa argentina. "
    "Respondé ÚNICAMENTE con un valor JSON válido (sin prosa fuera del "
    "JSON). Si un dato no aparece en el texto, omitilo o usá null; nunca "
    "inventes datos."
)


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=_TEXT_MAX)
    model: str | None = Field(default=None, max_length=200)
    on_behalf_of: str | None = Field(default=None, max_length=200)
    """Usuario de la plataforma llamadora en cuyo nombre se ejecuta (auditoría)."""


class ActaIn(TextIn):
    kind: str | None = Field(default=None, max_length=60)
    """acta_directorio | acta_asamblea (ayuda al prompt; opcional)."""


class WorkflowIn(TextIn):
    workflow: str = Field(min_length=1, max_length=60)
    """contract_review | acta_check | doc_summary (u otros futuros)."""


class SearchDoc(BaseModel):
    id: str = Field(max_length=64)
    text: str = Field(max_length=8_000)


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    documents: list[SearchDoc] = Field(default_factory=list, max_length=100)
    limit: int = Field(default=3, ge=1, le=20)
    model: str | None = Field(default=None, max_length=200)
    on_behalf_of: str | None = Field(default=None, max_length=200)
    """Usuario de la plataforma llamadora en cuyo nombre se ejecuta (auditoría)."""


class AdapterOut(BaseModel):
    payload: Any
    request_id: str


def _parse_json_payload(raw_text: str) -> Any:
    """Extract the JSON value from the model output (fenced or bare)."""
    candidate = raw_text.strip()
    match = _CODE_FENCE_RE.search(raw_text)
    if match is not None:
        candidate = match.group(1)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # último intento: primer bloque que parezca JSON (array u objeto)
        for opener, closer in (("[", "]"), ("{", "}")):
            start, end = candidate.find(opener), candidate.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    continue
    raise HTTPException(
        status_code=502,
        detail="El modelo no devolvió JSON parseable",
    )


async def _complete(
    gateway: GatewayClient,
    *,
    db: AsyncSession,
    user_id: str,
    action: str,
    on_behalf_of: str | None,
    system: str,
    user_content: str,
    model: str | None,
) -> AdapterOut:
    request_id = str(uuid.uuid4())
    gw_request = ChatCompletionRequest(
        model=model or DEFAULT_MODEL_ALIAS,
        messages=[
            ChatCompletionMessage(role="system", content=f"{_SYSTEM_PREAMBLE}\n\n{system}"),
            ChatCompletionMessage(role="user", content=user_content),
        ],
        stream=False,
        temperature=0.0,
        lq_ai_user_id=user_id,
    )
    response = await gateway.chat_completion(gw_request, request_id=request_id)

    # Auditar acá y no después de parsear: la inferencia ya corrió y ya se
    # pagó, así que una salida impresentable (el 502 de abajo) tiene que
    # dejar rastro igual.
    row = await audit_action(
        db,
        user_id=uuid.UUID(user_id),
        action=action,
        resource_type="legvolution",
        routed_inference_tier=response.routed_inference_tier,
        routed_provider=response.routed_provider,
        details={"on_behalf_of": on_behalf_of, "request_id": request_id},
    )
    # El request_id lo generamos nosotros, no viene en un header: el helper
    # no puede leerlo, así que se completa la columna a mano. Es por donde
    # se joinea contra el inference_routing_log del gateway.
    row.request_id = request_id
    # audit_action inserta y flushea pero no commitea; el adaptador no tiene
    # otra transacción en vuelo, así que el commit va acá o la fila se pierde.
    await db.commit()

    raw = response.choices[0].message.content if response.choices else ""
    payload = _parse_json_payload(raw if isinstance(raw, str) else "")
    return AdapterOut(payload=payload, request_id=request_id)


@router.post("/summarize-contract", response_model=AdapterOut)
async def summarize_contract(
    body: TextIn,
    user: ActiveUser,
    gateway: Annotated[GatewayClient, Depends(get_gateway_client)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AdapterOut:
    """Términos clave de un contrato: ``[{key, label, value}]``."""
    system = (
        "Extraé los términos clave del contrato. Devolvé un array JSON de "
        'objetos {"key": str, "label": str, "value": str}. Keys '
        "esperadas cuando apliquen: contraparte, objeto, montos, plazo, "
        "renovacion, preaviso, ley, jurisdiccion, confidencialidad, "
        "exclusividad, garantias. label en español legible; value conciso "
        "citando el dato del texto. Máximo 12 términos."
    )
    return await _complete(
        gateway,
        db=db,
        user_id=str(user.id),
        action="legvolution.summarize-contract",
        on_behalf_of=body.on_behalf_of,
        system=system,
        user_content=body.text,
        model=body.model,
    )


@router.post("/extract-obligations", response_model=AdapterOut)
async def extract_obligations(
    body: TextIn,
    user: ActiveUser,
    gateway: Annotated[GatewayClient, Depends(get_gateway_client)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AdapterOut:
    """Obligaciones y vencimientos: ``[{description, due_date, label}]``."""
    system = (
        "Identificá obligaciones de seguimiento y vencimientos del contrato "
        "(pagos, entregables, preavisos, renovaciones, vencimiento). Devolvé "
        'un array JSON de objetos {"description": str, "due_date": '
        '"YYYY-MM-DD" | null, "label": str}. label es una categoría '
        "corta (Vencimiento, Preaviso, Pagos, Entrega, Otro). Máximo 8."
    )
    return await _complete(
        gateway,
        db=db,
        user_id=str(user.id),
        action="legvolution.extract-obligations",
        on_behalf_of=body.on_behalf_of,
        system=system,
        user_content=body.text,
        model=body.model,
    )


@router.post("/extract-appointments", response_model=AdapterOut)
async def extract_appointments(
    body: ActaIn,
    user: ActiveUser,
    gateway: Annotated[GatewayClient, Depends(get_gateway_client)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AdapterOut:
    """Movimientos de autoridades en un acta: altas y ceses.

    ``[{person_name, role, title, action, since, until}]`` con
    ``role ∈ director|syndic|manager|attorney`` y ``action ∈ alta|cese``.
    """
    kind_hint = f"El documento es un {body.kind}. " if body.kind else ""
    system = (
        f"{kind_hint}Analizá el acta societaria y extraé los movimientos de "
        'autoridades. Devolvé un array JSON de objetos {"person_name": str, '
        '"role": "director"|"syndic"|"manager"|"attorney", '
        '"title": str (p. ej. "Presidente", "Director Titular"), '
        '"action": "alta"|"cese", "since": "YYYY-MM-DD" | null, '
        '"until": "YYYY-MM-DD" | null}. Para designaciones usá action '
        '"alta" con since = fecha del acta y until = fin del mandato si '
        'se menciona ("por N ejercicios", "con mandato hasta ..."). '
        "Para renuncias aceptadas, remociones o reemplazos usá action "
        '"cese" con until = fecha del acta.'
    )
    return await _complete(
        gateway,
        db=db,
        user_id=str(user.id),
        action="legvolution.extract-appointments",
        on_behalf_of=body.on_behalf_of,
        system=system,
        user_content=body.text,
        model=body.model,
    )


@router.post("/check-acta", response_model=AdapterOut)
async def check_acta(
    body: ActaIn,
    user: ActiveUser,
    gateway: Annotated[GatewayClient, Depends(get_gateway_client)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AdapterOut:
    """Checklist formal de un acta: ``[{key, severity, label, detail}]``."""
    kind_hint = f"El documento es un {body.kind}. " if body.kind else ""
    system = (
        f"{kind_hint}Revisá el acta societaria (derecho argentino, LGS "
        "19.550) como checklist formal. Devolvé un array JSON de objetos "
        '{"key": str, "severity": "ok"|"warn", "label": str, '
        '"detail": str}. Cubrí: fecha de celebración, quién preside, '
        "quórum (capital presente en asamblea / mayoría en directorio), "
        "orden del día, convocatoria (1ª/2ª), resoluciones claras, firmas. "
        'severity "warn" cuando el punto falta o es dudoso; "ok" cuando '
        "está correcto. detail en español, citando el texto cuando ayude."
    )
    return await _complete(
        gateway,
        db=db,
        user_id=str(user.id),
        action="legvolution.check-acta",
        on_behalf_of=body.on_behalf_of,
        system=system,
        user_content=body.text,
        model=body.model,
    )


_WORKFLOW_PROMPTS: dict[str, str] = {
    "contract_review": (
        "Hacé una revisión de contrato. Devolvé un objeto JSON "
        '{"summary": str (2-4 oraciones), "terms": [{"key": str, '
        '"label": str, "value": str}], "risks": [{"severity": '
        '"alta"|"media"|"baja", "label": str, "detail": str}], '
        '"obligations": [{"description": str, "due_date": '
        '"YYYY-MM-DD" | null, "label": str}]}.'
    ),
    "acta_check": (
        "Revisá el acta societaria como checklist formal (LGS 19.550). "
        'Devolvé un objeto JSON {"findings": [{"key": str, '
        '"severity": "ok"|"warn", "label": str, "detail": str}]}.'
    ),
    "doc_summary": (
        "Resumí el documento legal para un abogado interno. Devolvé un "
        'objeto JSON {"summary": str (5-10 oraciones), "key_points": '
        "[str] (máximo 8)}."
    ),
}


@router.post("/run-workflow", response_model=AdapterOut)
async def run_workflow(
    body: WorkflowIn,
    user: ActiveUser,
    gateway: Annotated[GatewayClient, Depends(get_gateway_client)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AdapterOut:
    """Ejecuta un workflow del catálogo del workbench legal."""
    system = _WORKFLOW_PROMPTS.get(body.workflow)
    if system is None:
        raise HTTPException(
            status_code=422,
            detail=f"Workflow desconocido: {body.workflow}",
        )
    return await _complete(
        gateway,
        db=db,
        user_id=str(user.id),
        action="legvolution.run-workflow",
        on_behalf_of=body.on_behalf_of,
        system=system,
        user_content=body.text,
        model=body.model,
    )


@router.post("/semantic-search", response_model=AdapterOut)
async def semantic_search(
    body: SearchIn,
    user: ActiveUser,
    gateway: Annotated[GatewayClient, Depends(get_gateway_client)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AdapterOut:
    """Ranking semántico de documentos candidatos frente a una consulta.

    ``[{id, score, snippet}]`` ordenado de mayor a menor relevancia.
    """
    if not body.documents:
        return AdapterOut(payload=[], request_id=str(uuid.uuid4()))
    system = (
        "Compará la consulta con los documentos candidatos y devolvé un "
        'array JSON de objetos {"id": str, "score": float entre 0 y 1, '
        '"snippet": str (fragmento relevante del documento)} SOLO con los '
        f"documentos realmente relacionados (máximo {body.limit}), ordenado "
        "por score descendente. Si ninguno es relevante devolvé []."
    )
    docs_block = "\n\n".join(f"[id: {d.id}]\n{d.text}" for d in body.documents)
    user_content = f"Consulta:\n{body.query}\n\nDocumentos candidatos:\n\n{docs_block}"
    return await _complete(
        gateway,
        db=db,
        user_id=str(user.id),
        action="legvolution.semantic-search",
        on_behalf_of=body.on_behalf_of,
        system=system,
        user_content=user_content,
        model=body.model,
    )
