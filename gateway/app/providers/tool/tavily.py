"""``tavily`` tool provider — commercial web-search egress (ADR 0014).

Single read-only tool: ``search_web`` over ``POST {base_url}/search`` with
``Authorization: Bearer <API key>`` (Tavily has no dedicated health endpoint,
so ``health_check`` probes ``/search`` with ``max_results=1``). Every outbound
call passes ``validate_egress_target`` (SSRF) per ADR 0014 D2; errors are
public-safe and key-scrubbed. Web results are public context, so they are
marked ``skip_anonymization=True`` for verbatim delivery (ADR 0014 D5) — the
same posture as the govinfo/eurlex public-text adapters.

NOTE: web context is NOT authoritative. The tool description tells the model
to prefer statutory/regulatory sources (govinfo, eurlex, courtlistener) and
to use ``include_domains`` (e.g. ``boletinoficial.gob.ar``) to scope results.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import ToolProviderConfig
from app.providers.base import ProviderHealth
from app.providers.tool.base import (
    ToolProviderAdapter,
    ToolProviderAuthError,
    ToolProviderError,
    ToolProviderHTTPError,
    ToolProviderInvalidRequestError,
    ToolProviderNetworkError,
    ToolResult,
    ToolSpec,
)
from app.providers.tool.egress import EgressRefused, validate_egress_target
from app.secrets import ProviderKeyResolver

DEFAULT_TIMEOUT_SECONDS = 30.0

# Tavily accepts up to 20 results per request; we also cap include_domains.
_MAX_RESULTS = 20
_DEFAULT_RESULTS = 8
_MAX_INCLUDE_DOMAINS = 10


class TavilyToolAdapter(ToolProviderAdapter):
    """Tool adapter for the Tavily Search API (api.tavily.com).

    Auth: ``Authorization: Bearer <key>`` header. One read-only operation,
    ``search_web``, normalizing Tavily responses to
    ``{"query", "results": [{"title", "url", "snippet", "score"}], "count"}``
    (Tavily's ``content`` field becomes ``snippet``).
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        allowlist: list[str],
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._allowlist = allowlist
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)

    @classmethod
    def from_config(
        cls,
        provider: ToolProviderConfig,
        *,
        key_resolver: ProviderKeyResolver | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> TavilyToolAdapter:
        if provider.type != "tavily":
            raise ValueError(f"TavilyToolAdapter from non-tavily provider {provider.type!r}")
        resolver = key_resolver or ProviderKeyResolver.from_environ()
        api_key = resolver.resolve(
            provider_name=provider.name,
            api_key_env=provider.api_key_env,
            api_key_encrypted=provider.api_key_encrypted,
        )
        if not api_key:
            raise ValueError(
                f"Tool provider {provider.name!r}: no Tavily API key resolved "
                f"(set {provider.api_key_env or 'TAVILY_API_KEY'})."
            )
        return cls(
            name=provider.name,
            base_url=provider.base_url,
            api_key=api_key,
            allowlist=provider.allowlist.hosts,
            client=client,
        )

    def validate_base_url(self) -> None:
        validate_egress_target(self._base_url + "/", allowlist=self._allowlist)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """SSRF-guard + issue one request with Bearer auth; map errors."""
        url = f"{self._base_url}{path}"
        validate_egress_target(url, allowlist=self._allowlist)
        headers: dict[str, str] = {"Authorization": f"Bearer {self._api_key}"}
        try:
            resp = await self._client.request(method, url, json=json_body, headers=headers)
        except EgressRefused:
            raise
        except httpx.HTTPError as exc:
            raise ToolProviderNetworkError(f"tavily network error: {exc}") from exc
        if resp.status_code in (401, 403):
            raise ToolProviderAuthError("tavily rejected the API key")
        if resp.status_code == 429:
            raise ToolProviderHTTPError("tavily rate limit", upstream_status=429)
        if 400 <= resp.status_code < 500:
            raise ToolProviderInvalidRequestError(
                f"tavily rejected the request ({resp.status_code})",
                upstream_status=resp.status_code,
            )
        if resp.status_code >= 500:
            raise ToolProviderHTTPError("tavily upstream error", upstream_status=resp.status_code)
        return resp

    async def list_tools(self, *, user_token: str | None = None) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="search_web",
                description=(
                    "General web search via Tavily. Returns titles, URLs, snippets, and "
                    "relevance scores from the public web. Context returned is NOT "
                    "authoritative and does not replace legal sources: verify every "
                    "finding against primary authorities (e.g. govinfo, eurlex, "
                    "courtlistener tools). Prefer 'include_domains' to scope results to "
                    "official sites (e.g. boletinoficial.gob.ar, infoleg.gob.ar)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string (non-empty).",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": (
                                f"Maximum results to return, 1-{_MAX_RESULTS} "
                                f"(default {_DEFAULT_RESULTS})."
                            ),
                        },
                        "include_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Restrict results to these domains (max "
                                f"{_MAX_INCLUDE_DOMAINS}), e.g. "
                                '["boletinoficial.gob.ar"].'
                            ),
                        },
                    },
                    "required": ["query"],
                },
                read_only=True,
            ),
        ]

    async def invoke_tool(
        self, tool: str, args: dict[str, Any], *, request_id: str, user_token: str | None = None
    ) -> ToolResult:
        if tool == "search_web":
            return await self._search_web(args)
        raise ToolProviderError(f"unknown tool {tool!r} for tavily provider")

    async def _search_web(self, args: dict[str, Any]) -> ToolResult:
        """POST /search — normalize Tavily results for the tool-loop.

        Validates that ``query`` is a non-empty string; clamps ``max_results``
        to 1..20 (default 8) and ``include_domains`` to at most 10 entries.
        Raises :class:`~app.providers.tool.base.ToolProviderInvalidRequestError`
        on bad input so the router can log a clean refusal.
        """
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolProviderInvalidRequestError(
                "search_web: 'query' must be a non-empty string",
                upstream_status=400,
            )
        raw_max = args.get("max_results", _DEFAULT_RESULTS)
        max_results: int = (
            raw_max
            if isinstance(raw_max, int)
            and not isinstance(raw_max, bool)
            and 1 <= raw_max <= _MAX_RESULTS
            else _DEFAULT_RESULTS
        )

        body: dict[str, Any] = {"query": query.strip(), "max_results": max_results}
        include_domains = args.get("include_domains")
        if isinstance(include_domains, list) and include_domains:
            body["include_domains"] = [str(d) for d in include_domains][:_MAX_INCLUDE_DOMAINS]

        resp = await self._request("POST", "/search", json_body=body)
        data: dict[str, Any] = resp.json()

        results = [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": r.get("content"),
                "score": r.get("score"),
            }
            for r in data.get("results", [])
        ]
        payload: dict[str, Any] = {
            "query": body["query"],
            "results": results,
            "count": len(results),
        }
        return self._result("search_web", payload, sent=body, received=data)

    def _result(
        self,
        tool: str,
        payload: Any,
        *,
        sent: Any,
        received: Any,
    ) -> ToolResult:
        """Build a ToolResult with byte counts; mark public web text verbatim.

        Same provenance discipline as :class:`GovInfoToolAdapter._result`:
        ``bytes_out``/``bytes_in`` over the JSON bodies, and
        ``skip_anonymization=True`` because Tavily returns public web content
        (ADR 0014 D5).
        """
        return ToolResult(
            provider=self.name,
            tool=tool,
            payload=payload,
            bytes_out=len(json.dumps(sent).encode("utf-8")),
            bytes_in=len(json.dumps(received).encode("utf-8")),
            skip_anonymization=True,
        )

    async def health_check(self) -> ProviderHealth:
        # Tavily has no dedicated health endpoint; a 1-result /search probe
        # is the cheapest honest reachability+credential check.
        try:
            await self._request("POST", "/search", json_body={"query": "ping", "max_results": 1})
        except ToolProviderError as exc:
            return ProviderHealth(name=self.name, reachable=False, error=str(exc))
        return ProviderHealth(name=self.name, reachable=True, latency_ms=0)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
