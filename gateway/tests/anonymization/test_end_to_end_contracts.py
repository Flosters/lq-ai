"""Contratos por el middleware completo, con el proveedor interceptado.

Lo que se verifica acá y no se puede verificar en los tests de unidad: que lo
que *sale* del gateway hacia el proveedor no tiene datos personales, y que lo
que *vuelve* al caller sí los tiene. No se intercepta nada por red — se llama
al middleware directamente y se inspecciona el request ya mutado, que es
exactamente el objeto que el adaptador va a serializar.
"""

import json

import pytest

from app.anonymization.engine import Anonymizer, _reset_analyzer_engine_for_tests
from app.anonymization.middleware import pre_anonymize_request
from app.config import AnonymizationConfig
from app.providers.openai_schema import ChatCompletionMessage, ChatCompletionRequest
from tests.anonymization.test_spanish_corpus import (
    CONTRATO_COMPARECENCIA,
    CONTRATO_PAGOS,
)

# Datos personales que NO pueden aparecer en el body que va al proveedor.
# Cada uno está en al menos uno de los dos fragmentos del corpus.
DATOS_SENSIBLES = (
    "Mariana Beatriz Quiroga",
    "Ignacio Ferreyra Alcorta",
    "30-71234567-1",
    "27.345.678",
    "0170099255000000000123",
)


@pytest.fixture
def config() -> AnonymizationConfig:
    """Config con la capa prendida para un tier con egreso.

    Sin ``languages``: ``pre_anonymize_request`` no lo lee — el conjunto de
    idiomas vive en el ``Anonymizer`` que se le pasa al lado. Ponerlo acá
    sugeriría que la config maneja el motor, que no es el caso.
    """

    return AnonymizationConfig(enabled=True, apply_at_tiers=[3, 4, 5])


@pytest.fixture
def anonymizer() -> Anonymizer:
    _reset_analyzer_engine_for_tests()
    return Anonymizer(languages=("es", "en"))


@pytest.mark.slow
@pytest.mark.parametrize(
    "contrato", [CONTRATO_COMPARECENCIA, CONTRATO_PAGOS], ids=["comparecencia", "pagos"]
)
def test_provider_never_sees_personal_data(
    config: AnonymizationConfig, anonymizer: Anonymizer, contrato: str
) -> None:
    """El body que sale hacia el proveedor no tiene datos personales.

    Se serializa el request entero, no sólo el contenido de los mensajes: si
    un dato se filtrara por un campo que el pre-pass no toca, este test lo ve.
    """

    request = ChatCompletionRequest(
        model="fast",
        messages=[
            ChatCompletionMessage(role="system", content="Resumí el contrato."),
            ChatCompletionMessage(role="user", content=contrato),
        ],
    )

    mapper = pre_anonymize_request(
        chat_request=request,
        config=config,
        routed_tier=4,
        anonymizer=anonymizer,
    )
    assert mapper is not None, "el middleware no se disparó; revisá el config"

    saliente = json.dumps(request.model_dump(mode="json"), ensure_ascii=False)
    presentes = [dato for dato in DATOS_SENSIBLES if dato in contrato]
    assert presentes, "el fragmento no tiene ningún dato sensible; el test no prueba nada"

    filtrados = [dato for dato in presentes if dato in saliente]
    assert not filtrados, f"salieron datos personales en claro: {filtrados}"


@pytest.mark.slow
def test_caller_gets_originals_back_through_json(
    config: AnonymizationConfig, anonymizer: Anonymizer
) -> None:
    """La respuesta JSON vuelve al caller con los originales, y parsea."""

    request = ChatCompletionRequest(
        model="fast",
        messages=[ChatCompletionMessage(role="user", content=CONTRATO_COMPARECENCIA)],
        response_format={"type": "json_object"},
    )

    mapper = pre_anonymize_request(
        chat_request=request, config=config, routed_tier=4, anonymizer=anonymizer
    )
    assert mapper is not None

    # El "proveedor" contesta con los seudónimos que recibió.
    seudonimos = sorted(mapper.reverse(), key=len, reverse=True)[:2]
    originales = [mapper.reverse()[p] for p in seudonimos]
    respuesta_cruda = json.dumps(
        {"partes": seudonimos, "resumen": f"Contrato entre {seudonimos[0]} y otros."}
    )

    rehidratado = anonymizer.rehydrate(respuesta_cruda, mapper, json_safe=True)

    # Parsea — ese es el punto del fix de la Task 6.
    parseado = json.loads(rehidratado)

    # Y trae los originales de vuelta. Chequear sólo que los seudónimos no
    # están dejaría pasar una rehidratación que los borre en vez de
    # sustituirlos, así que se verifica el positivo.
    assert parseado["partes"] == originales
    assert originales[0] in parseado["resumen"]

    texto_final = json.dumps(parseado, ensure_ascii=False)
    for pseudonimo in seudonimos:
        assert pseudonimo not in texto_final


@pytest.mark.slow
def test_tier_1_local_skips_anonymization(
    config: AnonymizationConfig, anonymizer: Anonymizer
) -> None:
    """Tier 1 es local: no hay egreso, así que no se reescribe nada."""

    request = ChatCompletionRequest(
        model="fast",
        messages=[ChatCompletionMessage(role="user", content=CONTRATO_COMPARECENCIA)],
    )

    mapper = pre_anonymize_request(
        chat_request=request, config=config, routed_tier=1, anonymizer=anonymizer
    )

    assert mapper is None
    assert request.messages[0].content == CONTRATO_COMPARECENCIA
