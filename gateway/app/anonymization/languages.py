"""Idiomas soportados por la capa de anonimización y su modelo de spaCy.

Un solo lugar donde vive el "qué modelo para qué idioma", para que agregar
un idioma sea una línea acá más un ``spacy download`` en los Dockerfile, y
no una búsqueda por todo el módulo.

Los tamaños de modelo son una decisión de compromiso. ``en_core_web_lg`` ya
estaba desde M2-B2. Para español se arranca con ``md`` (~40MB) en lugar de
``lg`` (~570MB): es el que usa la config multilingüe de referencia de
Presidio, y subir de tamaño sin medir recall antes es gastar 570MB de imagen
a ciegas. El corpus de ``tests/anonymization/test_spanish_corpus.py`` es la
evidencia con la que se decide si hace falta subir.
"""

from __future__ import annotations

SPACY_MODELS: dict[str, str] = {
    "en": "en_core_web_lg",
    "es": "es_core_news_md",
}

DEFAULT_LANGUAGES: tuple[str, ...] = ("es", "en")
"""Español primero porque es el idioma del corpus de LegVolution; el orden
sólo afecta el orden en que se prueban los candidatos en ``detect_language``,
no el resultado del análisis — sólo se analiza contra el idioma elegido.
"""
