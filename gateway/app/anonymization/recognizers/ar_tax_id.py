"""CUIT / CUIL — identificador tributario y laboral argentino.

Once dígitos con dígito verificador módulo 11, habitualmente escritos
``XX-XXXXXXXX-X``. Es el identificador de mayor valor de este conjunto: un
CUIT identifica de manera única a una persona humana o jurídica ante AFIP,
así que es dato personal de los que la Ley 25.326 protege sin discusión.

Dos anclajes hacen que el falso positivo sea prácticamente cero:

* **La etiqueta.** Se exige ``CUIT`` / ``CUIL`` (con o sin puntos) cerca del
  número. Sin etiqueta no se matchea: once dígitos sueltos en un contrato son
  igual de probables como importe, número de expediente o código interno.
* **El dígito verificador.** Se valida la aritmética. Una cadena con la forma
  correcta y el verificador equivocado se descarta, que es lo que separa esto
  de una regex de once dígitos.

No se cubre el CDI ni el CIE (identificadores de extranjeros sin CUIT): son
raros en contratos corporativos y comparten forma con el CUIT, así que
entrarían por el mismo patrón si alguna vez hacen falta.
"""

from __future__ import annotations

import re

from presidio_analyzer import Pattern, PatternRecognizer

# Pesos del módulo 11, aplicados a los primeros diez dígitos.
_PESOS = (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)

# Prefijos de tipo que AFIP asigna. Filtra "00000000000" y otras cadenas que
# validan la aritmética por casualidad.
_PREFIJOS_VALIDOS = frozenset({"20", "23", "24", "27", "30", "33", "34"})


def validar_cuit(digitos: str) -> bool:
    """``True`` si ``digitos`` son once dígitos con verificador correcto."""

    if len(digitos) != 11 or not digitos.isdigit():
        return False
    if digitos[:2] not in _PREFIJOS_VALIDOS:
        return False

    total = sum(int(d) * p for d, p in zip(digitos[:10], _PESOS, strict=True))
    resto = total % 11
    verificador = 11 - resto
    if verificador == 11:
        verificador = 0
    elif verificador == 10:
        verificador = 9
    return verificador == int(digitos[10])


# Etiqueta obligatoria + once dígitos con separadores opcionales. El
# ``(?i)`` no hace falta: el registry de Presidio aplica IGNORECASE global.
_CUIT_RE = (
    r"(?:C\.?U\.?I\.?[TL]\.?)\s*"
    r"(?:N[°º]?\s*)?:?\s*"
    r"(\d{2}[-\s]?\d{8}[-\s]?\d)"
)

_SOLO_DIGITOS_RE = re.compile(r"\D")


class ArTaxIdRecognizer(PatternRecognizer):
    """Reconoce CUIT y CUIL como entidad ``AR_TAX_ID``.

    El patrón matchea etiqueta + número, pero ``validate_result`` recorta el
    veredicto al verificador: Presidio llama a ese hook con el texto del span
    y un ``False`` explícito descarta el hallazgo.
    """

    ENTITY = "AR_TAX_ID"

    def __init__(self, supported_language: str = "es") -> None:
        patterns = [
            Pattern(name="cuit_con_etiqueta", regex=_CUIT_RE, score=0.95),
        ]
        super().__init__(
            supported_entity=self.ENTITY,
            name="ArTaxIdRecognizer",
            patterns=patterns,
            context=["cuit", "cuil", "afip", "inscripta", "inscripto"],
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        """Descarta el hallazgo si el dígito verificador no cierra."""

        digitos = _SOLO_DIGITOS_RE.sub("", pattern_text)
        return validar_cuit(digitos)
