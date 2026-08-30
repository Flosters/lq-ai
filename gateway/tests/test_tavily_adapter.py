"""Tests for the Tavily web-search tool-provider adapter (Task 8).

Mirrors the harness of ``tests/test_govinfo_adapter.py``: ``respx.mock``
intercepts outbound HTTP, DNS resolution is stubbed so the SSRF guard's
public-IP check passes offline, and the adapter is built via
``from_config`` with an env-var API key.
"""

import httpx
import pytest
import respx

from app.config import ToolProviderConfig
from app.providers.tool.tavily import TavilyToolAdapter

_TAVILY_CFG_DICT = {
    "name": "tavily-prod",
    "type": "tavily",
    "base_url": "https://api.tavily.com",
    "api_key_env": "TAVILY_API_KEY",
    "egress_tier": 4,
    "allowlist": {"hosts": ["api.tavily.com"]},
    "rate_limit": {"requests_per_minute": 60},
}

_TAVILY_OK = {
    "query": "ley 24.240 defensa del consumidor",
    "results": [
        {
            "title": "Ley 24.240",
            "url": "https://www.boletinoficial.gob.ar/ley24240",
            "content": "Texto de la ley de defensa del consumidor.",
            "score": 0.9,
        }
    ],
}


def _cfg() -> ToolProviderConfig:
    return ToolProviderConfig.model_validate(_TAVILY_CFG_DICT)


def _adapter(monkeypatch) -> TavilyToolAdapter:
    monkeypatch.setattr("app.providers.tool.egress._resolve_ips", lambda host: ["93.184.216.34"])
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    return TavilyToolAdapter.from_config(_cfg())


@pytest.mark.asyncio
async def test_from_config_type_check():
    """TavilyToolAdapter.from_config rejects non-tavily provider configs."""
    cfg = ToolProviderConfig.model_validate(
        {
            "name": "x",
            "type": "courtlistener",
            "base_url": "https://www.courtlistener.com",
            "api_key_env": "COURTLISTENER_API_TOKEN",
            "egress_tier": 4,
            "allowlist": {"hosts": ["www.courtlistener.com"]},
            "rate_limit": {"requests_per_minute": 60},
        }
    )
    with pytest.raises(ValueError):
        TavilyToolAdapter.from_config(cfg)


@pytest.mark.asyncio
async def test_search_web_sends_bearer_header(monkeypatch):
    adapter = _adapter(monkeypatch)
    with respx.mock:
        route = respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json=_TAVILY_OK)
        )
        await adapter.invoke_tool(
            "search_web", {"query": "ley 24.240"}, request_id="r1"
        )
    assert route.calls.last.request.headers["Authorization"] == "Bearer tvly-test"


@pytest.mark.asyncio
async def test_search_web_normalizes_results(monkeypatch) -> None:
    """Tavily's ``content`` field maps to ``snippet``; count reflects results."""
    adapter = _adapter(monkeypatch)
    with respx.mock:
        respx.route(method__in=["GET", "POST"], host="api.tavily.com").mock(
            return_value=httpx.Response(200, json=_TAVILY_OK)
        )
        out = await adapter.invoke_tool(
            "search_web",
            {"query": "ley 24.240 defensa del consumidor"},
            request_id="r1",
        )
    assert out.provider == "tavily-prod"
    assert out.payload["query"] == "ley 24.240 defensa del consumidor"
    assert out.payload["count"] == 1
    assert out.payload["results"][0]["title"] == "Ley 24.240"
    assert out.payload["results"][0]["url"].startswith("https://")
    assert out.payload["results"][0]["snippet"] == "Texto de la ley de defensa del consumidor."
    assert out.skip_anonymization is True
    # Provenance byte counts are populated (parity with govinfo's _result).
    assert out.bytes_out > 0
    assert out.bytes_in > 0


@pytest.mark.asyncio
async def test_search_web_empty_query_raises(monkeypatch) -> None:
    """Empty / whitespace query → ToolProviderInvalidRequestError."""
    from app.providers.tool.base import ToolProviderInvalidRequestError

    adapter = _adapter(monkeypatch)
    with pytest.raises(ToolProviderInvalidRequestError):
        await adapter.invoke_tool("search_web", {"query": "   "}, request_id="r2")


@pytest.mark.asyncio
async def test_search_web_absent_query_raises(monkeypatch) -> None:
    """Absent 'query' key (not just blank) → ToolProviderInvalidRequestError."""
    from app.providers.tool.base import ToolProviderInvalidRequestError

    adapter = _adapter(monkeypatch)
    with pytest.raises(ToolProviderInvalidRequestError):
        await adapter.invoke_tool("search_web", {}, request_id="r6")


@pytest.mark.asyncio
async def test_search_web_unknown_tool_raises(monkeypatch):
    from app.providers.tool.base import ToolProviderError

    adapter = _adapter(monkeypatch)
    with pytest.raises(ToolProviderError):
        await adapter.invoke_tool("nope", {}, request_id="r7")


@pytest.mark.asyncio
async def test_api_key_invalida_mapea_a_auth_error(monkeypatch) -> None:
    """401 from Tavily → ToolProviderAuthError (public-safe message)."""
    from app.providers.tool.base import ToolProviderAuthError

    adapter = _adapter(monkeypatch)
    with respx.mock:
        respx.route(method__in=["GET", "POST"], host="api.tavily.com").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid API key"})
        )
        with pytest.raises(ToolProviderAuthError):
            await adapter.invoke_tool("search_web", {"query": "x"}, request_id="r3")


@pytest.mark.asyncio
async def test_upstream_500_mapea_a_http_error(monkeypatch) -> None:
    """5xx from Tavily → ToolProviderHTTPError with upstream_status."""
    from app.providers.tool.base import ToolProviderHTTPError

    adapter = _adapter(monkeypatch)
    with respx.mock:
        respx.route(method__in=["GET", "POST"], host="api.tavily.com").mock(
            return_value=httpx.Response(503, json={"detail": "unavailable"})
        )
        with pytest.raises(ToolProviderHTTPError) as excinfo:
            await adapter.invoke_tool("search_web", {"query": "x"}, request_id="r8")
    assert excinfo.value.upstream_status == 503


@pytest.mark.asyncio
async def test_search_web_include_domains_forwarded(monkeypatch) -> None:
    """include_domains is passed through to the Tavily request body."""
    adapter = _adapter(monkeypatch)
    with respx.mock:
        route = respx.post("https://api.tavily.com/search").mock(
            return_value=httpx.Response(200, json={"query": "tasa", "results": []})
        )
        out = await adapter.invoke_tool(
            "search_web",
            {"query": "tasa de justicia", "include_domains": ["boletinoficial.gob.ar"]},
            request_id="r9",
        )
    import json as _json

    sent = _json.loads(route.calls.last.request.content)
    assert sent["include_domains"] == ["boletinoficial.gob.ar"]
    assert out.payload["count"] == 0


@pytest.mark.asyncio
async def test_list_tools_expone_search_web_readonly(monkeypatch) -> None:
    adapter = _adapter(monkeypatch)
    tools = await adapter.list_tools()
    assert [t.name for t in tools] == ["search_web"]
    assert tools[0].read_only is True
    assert tools[0].destructive is False
    assert tools[0].requires_confirmation is False
    # The description must disclaim non-authoritative web context.
    assert "NOT authoritative" in tools[0].description
    assert "does not replace" in tools[0].description


@pytest.mark.asyncio
async def test_health_check_reports_unreachable_on_error(monkeypatch) -> None:
    """Tavily has no dedicated health endpoint; a failing /search probe →
    ProviderHealth(reachable=False)."""
    from app.providers.base import ProviderHealth

    adapter = _adapter(monkeypatch)
    with respx.mock:
        respx.route(method__in=["GET", "POST"], host="api.tavily.com").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid API key"})
        )
        health = await adapter.health_check()
    assert isinstance(health, ProviderHealth)
    assert health.name == "tavily-prod"
    assert health.reachable is False
    assert health.error is not None


@pytest.mark.asyncio
async def test_health_check_reachable(monkeypatch) -> None:
    adapter = _adapter(monkeypatch)
    with respx.mock:
        respx.route(method__in=["GET", "POST"], host="api.tavily.com").mock(
            return_value=httpx.Response(200, json={"query": "ping", "results": []})
        )
        health = await adapter.health_check()
    assert health.reachable is True
