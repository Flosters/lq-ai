"""Detector de idioma — conteo de palabras funcionales, sesgado al español.

El sesgo no es estético, es la conclusión de una medición: un contrato
español analizado con el modelo inglés produce ocho falsos positivos
destructivos, y uno inglés analizado con el modelo español pierde un nombre
de sociedad. Errar hacia el español cuesta ~1/8 que errar hacia el inglés,
así que el inglés hay que ganárselo.
"""

import pytest

from app.anonymization.language_detect import detect_language

CANDIDATOS = ("es", "en")

ESPANOL = [
    "Entre GONZÁLEZ HERMANOS S.A. y TECNOLOGÍA DEL SUR S.R.L. se celebra el "
    "presente contrato de locación, en adelante LA LOCADORA y LA LOCATARIA.",
    "El Receptor se obliga a no divulgar la Información Confidencial que le "
    "sea entregada por el Emisor durante la vigencia de este acuerdo.",
    "La presente cláusula se rige por las leyes de la República Argentina y "
    "las partes se someten a los tribunales ordinarios de la Ciudad.",
]

INGLES = [
    "This Agreement is entered into by and between Acme Holdings Inc. and "
    "the Supplier, and shall be governed by the laws of the State of New York.",
    "The Receiving Party shall not disclose any Confidential Information "
    "furnished by the Disclosing Party during the term of this agreement.",
    "Notwithstanding the foregoing, either party may terminate this contract "
    "upon thirty days written notice to the other party.",
]


@pytest.mark.parametrize("texto", ESPANOL)
def test_detecta_espanol(texto):
    assert detect_language(texto, candidates=CANDIDATOS) == "es"


@pytest.mark.parametrize("texto", INGLES)
def test_detecta_ingles(texto):
    assert detect_language(texto, candidates=CANDIDATOS) == "en"


@pytest.mark.parametrize(
    "texto",
    [
        "",  # vacío
        "   \n\t  ",  # sólo espacios
        "30-71234567-1",  # sin palabras
        "OK",  # una palabra, ambigua
        "Anexo II",  # sin funcionales de ningún idioma
    ],
)
def test_ambiguo_cae_en_espanol(texto):
    """Sin evidencia suficiente gana el español: es el error barato."""
    assert detect_language(texto, candidates=CANDIDATOS) == "es"


def test_un_solo_candidato_no_analiza():
    """Con un idioma configurado, el detector es un paso trivial."""
    assert detect_language("This is clearly English", candidates=("es",)) == "es"
    assert detect_language("Esto es claramente español", candidates=("en",)) == "en"


def test_ingles_necesita_margen_no_solo_mayoria():
    """Texto mezclado con inglés apenas por delante sigue dando español.

    Una cláusula española que cita un término inglés ('the Agreement') no
    debe voltear la decisión — es el caso realista en contratos argentinos
    de operaciones cross-border.
    """
    mezcla = (
        "El presente contrato, en adelante the Agreement, se rige por las "
        "leyes de la República Argentina."
    )
    assert detect_language(mezcla, candidates=CANDIDATOS) == "es"


def test_candidato_desconocido_se_ignora():
    """Un idioma que el detector no sabe puntuar no puede ganar por default."""
    assert detect_language("the of and to the", candidates=("en", "de")) == "en"


def test_sin_candidatos_levanta():
    """Config rota se grita, no se degrada.

    Elegir un idioma por default con la lista vacía escondería un
    ``anonymization.languages`` mal configurado — en esta capa un error
    silencioso es una fuga, así que rompe fuerte.
    """
    with pytest.raises(ValueError, match="al menos un idioma"):
        detect_language("cualquier cosa", candidates=())


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("¿Qué obligaciones tiene la locadora?", "es"),
        ("What are the obligations of the lessor?", "en"),
    ],
)
def test_mensajes_cortos_de_chat(texto, esperado):
    """El caso flojo del conteo de funcionales: pocas palabras.

    Se verifica porque es el argumento principal a favor de usar una
    librería. Sobre preguntas de chat reales alcanza — y si alguna vez deja
    de alcanzar, este es el test que lo va a mostrar antes que un usuario.
    """
    assert detect_language(texto, candidates=("es", "en")) == esperado
