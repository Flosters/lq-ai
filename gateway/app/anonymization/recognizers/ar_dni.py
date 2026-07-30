"""DNI / LC / LE — documento de identidad argentino.

El más valioso de este conjunto y el único sin dígito verificador, lo que lo
vuelve el más difícil. ``27.345.678`` es tipográficamente indistinguible de
un importe en pesos, de un número de cláusula o de un código interno. Sin
nada que validar aritméticamente, el único anclaje honesto es la etiqueta.

**Decisión deliberada: no se matchean números sueltos.** Un DNI escrito sin
etiqueta ("el compareciente 27.345.678") no se detecta y sale en claro. La
alternativa —matchear siete u ocho dígitos por forma— convertiría todo
importe de un contrato en un seudónimo y haría el texto ilegible para el
modelo. La sub-detección es consistente con la postura conservadora del
resto de la capa (``docs/security/anonymization.md`` §Calibrando) y queda
anotada como límite conocido.

El span incluye la etiqueta, no sólo el número: ``DNI 27.345.678`` se
reemplaza entero por ``AR_DNI_0001``. Se pierde la señal de que ahí había un
documento, lo cual le quita un poco de contexto al modelo, pero evita el
lookbehind de ancho variable que ``re`` no soporta. La rehidratación
devuelve la etiqueta junto con el número, así que la respuesta al usuario
sale intacta.
"""

from __future__ import annotations

from presidio_analyzer import Pattern, PatternRecognizer

# Etiquetas que en la práctica preceden un documento en prosa jurídica
# argentina. ``L.C.`` (libreta cívica) y ``L.E.`` (libreta de enrolamiento)
# siguen apareciendo en comparecencias de personas mayores.
_ETIQUETA = (
    r"(?:D\.?N\.?I\.?"
    r"|documento(?:\s+nacional\s+de\s+identidad)?"
    r"|L\.?C\.?"
    r"|L\.?E\.?)"
)

# Ancla la etiqueta a un borde de palabra real. ``\b`` no alcanza: las
# alternativas de dos letras (``L.C.``, ``L.E.``) empiezan con una letra, y
# ``\b`` entre dos letras no es un borde — por eso "controLE", "detalLE",
# "alquiLE" y "CALLE" matcheaban antes de este fix (ver hallazgo de la
# revisión de rama: cualquier palabra terminada en "le"/"lc" seguida de un
# número de 7-8 dígitos se detectaba como DNI). El lookbehind negativo exige
# que lo que precede a la etiqueta no sea ni letra (con acentos/ñ) ni dígito.
_SIN_LETRA_ANTES = r"(?<![0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ])"

# Siete u ocho dígitos, con puntos de miles opcionales.
_NUMERO = r"(?:\d{1,2}\.\d{3}\.\d{3}|\d{7,8})"

# Lookahead negativo: sin esto, "DNI 123456789012" matchea sólo el prefijo
# "DNI 12345678" y deja "9012" en claro. Exige que el número no siga con
# otro dígito.
_SIN_DIGITO_DESPUES = r"(?!\d)"

_DNI_RE = rf"{_SIN_LETRA_ANTES}{_ETIQUETA}\s*(?:N[°º]?\s*)?:?\s*{_NUMERO}{_SIN_DIGITO_DESPUES}"


class ArDniRecognizer(PatternRecognizer):
    """Reconoce DNI/LC/LE como entidad ``AR_DNI``, anclado a la etiqueta."""

    ENTITY = "AR_DNI"

    def __init__(self, supported_language: str = "es") -> None:
        patterns = [
            Pattern(name="dni_con_etiqueta", regex=_DNI_RE, score=0.9),
        ]
        super().__init__(
            supported_entity=self.ENTITY,
            name="ArDniRecognizer",
            patterns=patterns,
            context=["dni", "documento", "identidad", "compareciente"],
            supported_language=supported_language,
        )
