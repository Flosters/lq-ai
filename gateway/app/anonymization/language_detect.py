"""Detección de idioma por palabras funcionales, sesgada al español.

Elige qué modelo de spaCy analiza un texto. Sólo afecta la mitad NER de la
capa (PERSON / ORGANIZATION / LOCATION); los reconocedores de patrón —email,
teléfono, CUIT, CUIL, CBU, DNI— se registran bajo todos los idiomas
configurados y disparan sin consultar esto. Un error del detector degrada el
recall de nombres y sociedades; nunca pierde un identificador.

Por qué a mano y no una librería
--------------------------------

El gateway es la frontera de egreso de la plataforma y el proyecto evita
dependencias que no sean estrictamente necesarias (ver los comentarios de
``pyproject.toml`` sobre no usar SDKs de LLM). Contar palabras funcionales
sobre prosa jurídica —larga, formal y densa en funcionales— es un problema
fácil que no justifica sumar superficie de supply-chain a este servicio. Si
alguna vez hace falta clasificar mensajes de chat de tres palabras, ahí sí
conviene una librería; para cuerpos de contrato, esto alcanza.

Por qué sesgado al español
--------------------------

Medición del 2026-07-30 sobre el corpus de contratos argentinos: analizar
español con ``en_core_web_lg`` produce ocho falsos positivos destructivos
(``'por'`` y ``'en adelante'`` como ORGANIZATION, ``'se celebra el
presente'`` como PERSON) y además pierde una de las dos sociedades.
Analizar inglés con ``es_core_news_md`` sólo pierde un nombre de sociedad,
sin generar basura. Los costos son asimétricos por cerca de 8 a 1, así que
el inglés tiene que ganar por margen, no por mayoría simple.
"""

from __future__ import annotations

import re

# Palabras funcionales de alta frecuencia y bajo solapamiento entre los dos
# idiomas. Se excluyen deliberadamente las que existen en ambos (``no``,
# ``a``, ``en`` es preposición española pero también aparece en inglés
# técnico) para que el conteo discrimine en vez de sumar ruido.
_FUNCIONALES: dict[str, frozenset[str]] = {
    "es": frozenset(
        {
            "de",
            "la",
            "el",
            "que",
            "los",
            "las",
            "del",
            "se",
            "por",
            "con",
            "para",
            "una",
            "como",
            "sus",
            "este",
            "esta",
            "presente",
            "entre",
            "adelante",
            "partes",
            "sera",
            "será",
            "podra",
            "podrá",
            "deberá",
            "cualquier",
            "conforme",
            "mismo",
            "dicha",
            "dicho",
        }
    ),
    "en": frozenset(
        {
            "the",
            "of",
            "and",
            "to",
            "shall",
            "any",
            "such",
            "this",
            "that",
            "with",
            "for",
            "which",
            "hereby",
            "herein",
            "thereof",
            "party",
            "parties",
            "agreement",
            "may",
            "not",
            "been",
            "under",
            "upon",
            "notwithstanding",
            "foregoing",
            "whereas",
        }
    ),
}

# El inglés tiene que superar al español por este factor para ganar. 1.5 sale
# de la asimetría medida: con paridad o ventaja chica conviene el español.
_MARGEN_INGLES = 1.5

_PALABRA_RE = re.compile(r"[a-záéíóúñü]+", re.IGNORECASE)


def detect_language(text: str, *, candidates: tuple[str, ...]) -> str:
    """Elegir de ``candidates`` el idioma con el que analizar ``text``.

    Con un solo candidato lo devuelve sin mirar el texto. Con varios cuenta
    palabras funcionales de cada idioma que sepa puntuar y aplica el sesgo:
    ``en`` gana sólo si supera a ``es`` por :data:`_MARGEN_INGLES`. Sin
    evidencia (texto vacío, sin palabras, empate) devuelve el primer
    candidato puntuable, que por configuración es ``es``.

    Un candidato para el que no haya lista de funcionales no puede ganar por
    conteo — sólo se usa como fallback si es el único.

    Levanta ``ValueError`` si ``candidates`` viene vacío. Es config rota, no
    un caso a degradar en silencio: elegir un idioma por default acá
    escondería un ``anonymization.languages`` mal configurado justo en la
    capa donde un error silencioso es una fuga.
    """

    if not candidates:
        raise ValueError(
            "detect_language necesita al menos un idioma candidato; "
            "revisá anonymization.languages en gateway.yaml."
        )

    if len(candidates) == 1:
        return candidates[0]

    puntuables = [c for c in candidates if c in _FUNCIONALES]
    if not puntuables:
        return candidates[0]
    if len(puntuables) == 1:
        return puntuables[0]

    palabras = [p.lower() for p in _PALABRA_RE.findall(text)]
    conteos = {
        idioma: sum(1 for p in palabras if p in _FUNCIONALES[idioma]) for idioma in puntuables
    }

    # El español es el default barato: se lo desbanca sólo con margen.
    if "es" in conteos and "en" in conteos:
        if conteos["en"] > conteos["es"] * _MARGEN_INGLES:
            return "en"
        return "es"

    # Sin el par es/en configurado, gana el conteo simple; empate → el
    # primer candidato puntuable, que respeta el orden de la config.
    mejor = max(puntuables, key=lambda idioma: conteos[idioma])
    if conteos[mejor] == 0:
        return puntuables[0]
    return mejor
