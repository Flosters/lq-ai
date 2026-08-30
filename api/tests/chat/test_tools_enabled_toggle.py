"""tools_enabled: None=auto (comportamiento actual), False=fuerza single-shot,
True=loop si hay tools configuradas. Fail-restrictive por defecto."""

import pytest
from pydantic import ValidationError

from app.schemas.chats import MessageCreateRequest


def test_default_es_none_auto():
    assert MessageCreateRequest(content="hola").tools_enabled is None


def test_acepta_true_y_false():
    assert MessageCreateRequest(content="h", tools_enabled=True).tools_enabled is True
    assert MessageCreateRequest(content="h", tools_enabled=False).tools_enabled is False


def test_rechaza_no_bool():
    with pytest.raises(ValidationError):
        MessageCreateRequest(content="h", tools_enabled="si")
