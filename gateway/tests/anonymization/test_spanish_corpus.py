"""Corpus de contratos argentinos — línea de base y verificación es+en.

El corpus es sintético a propósito: entidades inventadas, CUIT y CBU con
dígito verificador válido calculado a mano. No hay datos de ningún cliente
real acá, así que el archivo puede vivir en el repo sin problema.
"""

import pytest

from app.anonymization.engine import Anonymizer, _reset_analyzer_engine_for_tests
from app.anonymization.middleware import pre_anonymize_request
from app.config import AnonymizationConfig
from app.providers.openai_schema import ChatCompletionMessage, ChatCompletionRequest

# Fragmento 1: comparecencia típica de un contrato argentino.
#
# Los CUIT tienen dígito verificador válido — calculado con el mismo módulo 11
# que valida el reconocedor. Si los cambiás, recalculá el verificador o el
# reconocedor los va a descartar y los tests van a fallar por la razón
# equivocada.
CONTRATO_COMPARECENCIA = (
    "Entre GONZÁLEZ HERMANOS S.A., CUIT 30-71234567-1, con domicilio en "
    "Av. Corrientes 1234, Ciudad Autónoma de Buenos Aires, representada en "
    "este acto por Mariana Beatriz Quiroga, DNI 27.345.678, en adelante "
    '"LA LOCADORA", y TECNOLOGÍA DEL SUR S.R.L., CUIT 33-69876543-2, '
    'representada por Ignacio Ferreyra Alcorta, en adelante "LA LOCATARIA", '
    "se celebra el presente contrato."
)


CONTRATO_CONFIDENCIALIDAD = (
    "El Receptor se obliga a no divulgar la Información Confidencial. "
    "Las notificaciones se dirigirán a legales@gonzalezhermanos.com.ar o al "
    "teléfono +54 11 4321-8765, a la atención del Dr. Esteban Uriburu."
)

CONTRATO_PAGOS = (
    "LA LOCATARIA abonará el canon mediante transferencia al "
    "CBU 0170099255000000000123, titular GONZÁLEZ HERMANOS S.A., "
    "CUIT 30-71234567-1."
)

CONTRACT_ENGLISH = (
    "This Agreement is entered into by Acme Holdings Inc., a Delaware "
    "corporation, represented by Sarah Whitfield, whose counsel may be "
    "reached at swhitfield@acme-legal.com."
)


@pytest.mark.slow
def test_spanish_person_detected():
    """Los nombres de persona de un contrato argentino se seudonimizan."""
    _reset_analyzer_engine_for_tests()
    result = Anonymizer().pseudonymize(CONTRATO_COMPARECENCIA)

    assert "Mariana Beatriz Quiroga" not in result.text
    assert "Ignacio Ferreyra Alcorta" not in result.text
    assert "PERSON_" in result.text


@pytest.mark.slow
def test_spanish_argentine_identifiers_detected():
    """CUIT, CBU y DNI se seudonimizan en texto español."""
    _reset_analyzer_engine_for_tests()
    anonymizer = Anonymizer()

    comparecencia = anonymizer.pseudonymize(CONTRATO_COMPARECENCIA)
    assert "30-71234567-1" not in comparecencia.text
    assert "27.345.678" not in comparecencia.text

    pagos = anonymizer.pseudonymize(CONTRATO_PAGOS)
    assert "0170099255000000000123" not in pagos.text
    assert "30-71234567-1" not in pagos.text


@pytest.mark.slow
def test_pattern_recognizers_still_fire_in_spanish():
    """La regresión que este plan viene a evitar: mail y teléfono en español.

    Si el motor pasara a español a secas, los reconocedores de patrón —que se
    registran por idioma— dejarían de dispararse sin dar error. Este test lo
    hace ruidoso.
    """
    _reset_analyzer_engine_for_tests()
    result = Anonymizer().pseudonymize(CONTRATO_CONFIDENCIALIDAD)

    assert "legales@gonzalezhermanos.com.ar" not in result.text
    assert "EMAIL_ADDRESS_" in result.text


@pytest.mark.slow
def test_english_still_works():
    """El inglés sigue funcionando: el detector lo elige y el modelo inglés corre."""
    _reset_analyzer_engine_for_tests()
    result = Anonymizer().pseudonymize(CONTRACT_ENGLISH)

    assert "Sarah Whitfield" not in result.text
    assert "swhitfield@acme-legal.com" not in result.text


@pytest.mark.slow
def test_round_trip_byte_for_byte_spanish():
    """Seudonimizar y rehidratar devuelve el original exacto."""
    _reset_analyzer_engine_for_tests()
    anonymizer = Anonymizer()
    result = anonymizer.pseudonymize(CONTRATO_COMPARECENCIA)

    assert anonymizer.rehydrate(result.text, result.mapper) == CONTRATO_COMPARECENCIA


@pytest.mark.slow
def test_request_language_is_detected_once_for_all_messages():
    """Mensajes cortos en inglés van todos al mismo modelo.

    Es la invariante M2-C3 de estabilidad de seudónimos, que se rompía
    detectando por mensaje: ``"Discussing John Smith."`` no tiene ninguna
    palabra funcional, empataba 0-0, se iba al modelo español y devolvía un
    span distinto que el resto. Detectando sobre el request entero, las
    cuatro palabras funcionales inglesas de los otros mensajes arrastran a
    todo el request al modelo correcto.
    """
    _reset_analyzer_engine_for_tests()
    request = ChatCompletionRequest(
        model="smart",
        messages=[
            ChatCompletionMessage(role="system", content="Discussing John Smith."),
            ChatCompletionMessage(role="user", content="What did John Smith say?"),
            ChatCompletionMessage(role="assistant", content="John Smith said yes."),
        ],
    )

    mapper = pre_anonymize_request(
        chat_request=request,
        config=AnonymizationConfig(enabled=True, apply_at_tiers=[3, 4, 5], languages=["es", "en"]),
        routed_tier=4,
        anonymizer=Anonymizer(),
    )

    assert mapper is not None
    assert list(mapper.reverse().values()) == ["John Smith"], (
        "el mismo nombre en tres mensajes tiene que dar un solo seudónimo"
    )
    for message in request.messages:
        assert "John Smith" not in (message.content or "")
