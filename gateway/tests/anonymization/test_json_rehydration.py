"""Rehidratación dentro de una respuesta JSON.

Los adaptadores de LegVolution piden salida estructurada, así que el
``content`` que vuelve del proveedor es un JSON serializado. Rehidratar con
``str.replace`` mete el original crudo adentro de un string JSON: si el
original tiene comillas, barras o saltos de línea, el resultado no parsea.
"""

import json

import pytest

from app.anonymization.engine import Anonymizer
from app.anonymization.mapper import PseudonymMapper


@pytest.mark.parametrize(
    "original",
    [
        'Juan "Pepe" Pérez',  # comillas
        "Acme S.A.\nAv. Corrientes 1234\nCABA",  # multilínea
        "González \\ Hermanos",  # barra invertida
    ],
)
def test_rehydrate_json_safe_keeps_json_parseable(original):
    mapper = PseudonymMapper()
    pseudonym = mapper.assign("PERSON", original)
    respuesta = json.dumps({"parte": pseudonym, "rol": "locadora"})

    rehidratado = Anonymizer().rehydrate(respuesta, mapper, json_safe=True)

    # Parsea, y el valor recuperado es el original exacto.
    assert json.loads(rehidratado)["parte"] == original


@pytest.mark.parametrize(
    "original",
    ['Juan "Pepe" Pérez', "Acme S.A.\nAv. Corrientes 1234\nCABA"],
)
def test_rehydrate_without_json_safe_corrupts_json(original):
    """El bug, documentado. Sin el flag, el JSON queda inválido."""
    mapper = PseudonymMapper()
    pseudonym = mapper.assign("PERSON", original)
    respuesta = json.dumps({"parte": pseudonym})

    rehidratado = Anonymizer().rehydrate(respuesta, mapper, json_safe=False)

    with pytest.raises(json.JSONDecodeError):
        json.loads(rehidratado)


def test_rehydrate_prose_mode_unchanged():
    """Modo prosa: byte-for-byte, como siempre. No se rompe M2-C3."""
    mapper = PseudonymMapper()
    pseudonym = mapper.assign("PERSON", 'Juan "Pepe" Pérez')

    texto = f"El contrato lo firma {pseudonym}."
    assert Anonymizer().rehydrate(texto, mapper) == ('El contrato lo firma Juan "Pepe" Pérez.')
