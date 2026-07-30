"""Corpus de contratos argentinos — línea de base y verificación es+en.

El corpus es sintético a propósito: entidades inventadas, CUIT y CBU con
dígito verificador válido calculado a mano. No hay datos de ningún cliente
real acá, así que el archivo puede vivir en el repo sin problema.
"""

import pytest

from app.anonymization.engine import Anonymizer, _reset_analyzer_engine_for_tests

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
    "\"LA LOCADORA\", y TECNOLOGÍA DEL SUR S.R.L., CUIT 33-69876543-2, "
    "representada por Ignacio Ferreyra Alcorta, en adelante \"LA LOCATARIA\", "
    "se celebra el presente contrato."
)


@pytest.mark.slow
def test_baseline_TEMPORARY_english_engine_on_spanish_contract():
    """TEMPORAL — se borra en la Task 5 de este plan. No es un test de verdad.

    Afirma que el comportamiento roto *está* roto, para dejar en el repo la
    evidencia medida antes de tocar nada. Cubre los tres defectos reales del
    motor inglés sobre un contrato argentino — que NO son los que uno
    supondría, porque los nombres de persona sí se detectan:

    1. El DNI sale en claro (no hay reconocedor en ningún idioma).
    2. Una de las dos sociedades no se detecta (el modelo inglés la pierde).
    3. Prosa española común se come como entidad, corrompiendo el prompt:
       ``"se celebra el presente contrato"`` llega al proveedor como
       ``"PERSON_0003 contrato"``.

    Cuando las Tasks 3 y 5 estén hechas, este test va a fallar en las tres
    aserciones. Eso es el éxito, no una regresión: ahí se borra y lo
    reemplazan los tests positivos.

    El ``TEMPORARY`` en el nombre es a propósito — si aparece en un review
    después de la Task 5, alguien se olvidó de borrarlo.
    """
    _reset_analyzer_engine_for_tests()
    result = Anonymizer().pseudonymize(CONTRATO_COMPARECENCIA)

    # 1. Fuga real: el DNI sale tal cual.
    assert "27.345.678" in result.text

    # 2. Fuga real: el modelo inglés no ve la segunda sociedad.
    assert "TECNOLOGÍA DEL SUR S.R.L." in result.text

    # 3. Corrupción: la frase se comió como PERSON, así que ya no está entera.
    assert "se celebra el presente contrato" not in result.text
