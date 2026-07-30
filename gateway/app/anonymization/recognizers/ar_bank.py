"""CBU — Clave Bancaria Uniforme argentina.

Veintidós dígitos en dos bloques, cada uno con su propio dígito verificador
módulo 10. Aparece en cláusulas de pago, en cuentas de garantía y en anexos
de honorarios; identifica una cuenta bancaria concreta, así que es dato
personal o financiero según de quién sea la cuenta.

Mismo criterio que el CUIT: etiqueta obligatoria más verificación
aritmética. Los dos bloques se validan por separado porque así está definido
el estándar del BCRA — un CBU con el primer bloque bien y el segundo mal es
inválido, y esa es justamente la clase de cadena que una regex de veintidós
dígitos aceptaría.

No se cubre el CVU (billeteras virtuales). Comparte la longitud de 22
dígitos pero no el esquema de verificación, así que necesitaría su propio
patrón; se deja para cuando aparezca en un contrato de verdad.
"""

from __future__ import annotations

import re

from presidio_analyzer import Pattern, PatternRecognizer

_PESOS_BLOQUE_1 = (7, 1, 3, 9, 7, 1, 3)
_PESOS_BLOQUE_2 = (3, 9, 7, 1, 3, 9, 7, 1, 3, 9, 7, 1, 3)

_SOLO_DIGITOS_RE = re.compile(r"\D")


def _verificador_modulo_10(digitos: str, pesos: tuple[int, ...]) -> int:
    total = sum(int(d) * p for d, p in zip(digitos, pesos, strict=True))
    return (10 - (total % 10)) % 10


def validar_cbu(digitos: str) -> bool:
    """``True`` si ``digitos`` son 22 dígitos con los dos bloques válidos."""

    if len(digitos) != 22 or not digitos.isdigit():
        return False

    bloque_1, bloque_2 = digitos[:8], digitos[8:]
    if _verificador_modulo_10(bloque_1[:7], _PESOS_BLOQUE_1) != int(bloque_1[7]):
        return False
    return _verificador_modulo_10(bloque_2[:13], _PESOS_BLOQUE_2) == int(bloque_2[13])


_CBU_RE = (
    r"(?:C\.?B\.?U\.?|clave\s+bancaria(?:\s+uniforme)?)\s*"
    r"(?:N[°º]?\s*)?:?\s*"
    r"(\d{22}|\d{8}[-\s]\d{14})"
)


class ArBankRecognizer(PatternRecognizer):
    """Reconoce CBU como entidad ``AR_BANK_ACCOUNT``."""

    ENTITY = "AR_BANK_ACCOUNT"

    def __init__(self, supported_language: str = "es") -> None:
        patterns = [
            Pattern(name="cbu_con_etiqueta", regex=_CBU_RE, score=0.95),
        ]
        super().__init__(
            supported_entity=self.ENTITY,
            name="ArBankRecognizer",
            patterns=patterns,
            context=["cbu", "cuenta", "banco", "transferencia", "acreditación"],
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        return validar_cbu(_SOLO_DIGITOS_RE.sub("", pattern_text))
