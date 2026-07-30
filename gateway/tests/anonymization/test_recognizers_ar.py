"""Reconocedores de identificadores argentinos — positivos y negativos.

Los CUIT y CBU de los casos positivos tienen dígito verificador válido
(calculado con el mismo algoritmo que valida el reconocedor). Los negativos
son la parte que importa: cadenas con la forma correcta y el verificador
equivocado, que es exactamente lo que distingue a este reconocedor de una
regex de once dígitos.
"""

import pytest

from app.anonymization.recognizers.ar_tax_id import ArTaxIdRecognizer, validar_cuit


class TestValidarCuit:
    @pytest.mark.parametrize(
        "digitos",
        [
            "30712345671",  # persona jurídica (30-)
            "33698765432",  # persona jurídica (33-)
            "20123456786",  # persona humana masculina (20-)
        ],
    )
    def test_verificador_valido(self, digitos):
        assert validar_cuit(digitos) is True

    @pytest.mark.parametrize(
        "digitos",
        [
            "30712345672",  # último dígito corrido en uno
            "20123456780",
            "00000000000",  # el verificador cierra por casualidad, el prefijo no existe
        ],
    )
    def test_verificador_invalido(self, digitos):
        assert validar_cuit(digitos) is False

    @pytest.mark.parametrize("digitos", ["", "123", "307123456712"])
    def test_longitud_incorrecta(self, digitos):
        assert validar_cuit(digitos) is False


class TestArTaxIdRecognizer:
    @pytest.fixture
    def recognizer(self):
        return ArTaxIdRecognizer()

    @pytest.mark.parametrize(
        "texto,esperado",
        [
            ("CUIT 30-71234567-1 inscripta en IGJ", "30-71234567-1"),
            ("C.U.I.T. N° 30-71234567-1", "30-71234567-1"),
            ("CUIL 20-12345678-6 del trabajador", "20-12345678-6"),
            ("el cuit 30712345671 sin guiones", "30712345671"),
        ],
    )
    def test_detecta(self, recognizer, texto, esperado):
        results = recognizer.analyze(texto, entities=["AR_TAX_ID"])
        assert len(results) == 1, f"esperaba 1 hallazgo en {texto!r}"
        assert esperado in texto[results[0].start : results[0].end]

    @pytest.mark.parametrize(
        "texto",
        [
            "CUIT 30-71234567-2",           # verificador inválido
            "expediente 30-71234567-1",     # forma válida pero sin etiqueta CUIT
            "el importe fue de 30712345671 pesos",  # once dígitos sin etiqueta
            "factura 0001-00071234-1",      # otra cosa con guiones
        ],
    )
    def test_no_detecta(self, recognizer, texto):
        assert recognizer.analyze(texto, entities=["AR_TAX_ID"]) == []


from app.anonymization.recognizers.ar_bank import ArBankRecognizer, validar_cbu
from app.anonymization.recognizers.ar_dni import ArDniRecognizer


class TestValidarCbu:
    def test_verificador_valido(self):
        # Bloque 1: 0170099 + verificador 2. Bloque 2: 5500000000012 + 3.
        assert validar_cbu("0170099255000000000123") is True

    @pytest.mark.parametrize(
        "digitos",
        [
            "0170099155000000000123",  # verificador del bloque 1 roto
            "0170099255000000000124",  # verificador del bloque 2 roto
        ],
    )
    def test_verificador_invalido(self, digitos):
        assert validar_cbu(digitos) is False

    @pytest.mark.parametrize("digitos", ["", "017009925500000000012"])
    def test_longitud_incorrecta(self, digitos):
        assert validar_cbu(digitos) is False


class TestArBankRecognizer:
    @pytest.fixture
    def recognizer(self):
        return ArBankRecognizer()

    def test_detecta_con_etiqueta(self, recognizer):
        texto = "Los pagos se harán al CBU 0170099255000000000123 de LA LOCADORA."
        results = recognizer.analyze(texto, entities=["AR_BANK_ACCOUNT"])
        assert len(results) == 1
        assert "0170099255000000000123" in texto[results[0].start : results[0].end]

    @pytest.mark.parametrize(
        "texto",
        [
            "CBU 0170099255000000000124",         # verificador inválido
            "expediente 0170099255000000000123",  # sin etiqueta
        ],
    )
    def test_no_detecta(self, recognizer, texto):
        assert recognizer.analyze(texto, entities=["AR_BANK_ACCOUNT"]) == []


class TestArDniRecognizer:
    @pytest.fixture
    def recognizer(self):
        return ArDniRecognizer()

    @pytest.mark.parametrize(
        "texto",
        [
            "DNI 27.345.678",
            "D.N.I. N° 27345678",
            "documento nacional de identidad 27.345.678",
            "L.C. 4.567.890",
        ],
    )
    def test_detecta(self, recognizer, texto):
        results = recognizer.analyze(texto, entities=["AR_DNI"])
        assert len(results) == 1, f"esperaba 1 hallazgo en {texto!r}"

    @pytest.mark.parametrize(
        "texto",
        [
            "el precio es de $27.345.678 más IVA",  # importe: el caso que importa
            "la cláusula 27.345 del reglamento",
            "27345678",  # ocho dígitos sueltos, sin etiqueta
        ],
    )
    def test_no_detecta(self, recognizer, texto):
        assert recognizer.analyze(texto, entities=["AR_DNI"]) == []
