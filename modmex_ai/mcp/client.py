"""Transport client for MCP servers over JSON HTTP."""

from __future__ import annotations

import json
from typing import Any

from modmex_ai.http.async_client import AsyncHttpClient
from modmex_ai.http.sse import parse_sse_lines_async
from modmex_ai.mcp.tools import RemoteTool
from modmex_ai.mcp.errors import MCPError, MCPInputRequired
from modmex_ai.mcp.headers import extract_mcp_parameter_headers


class MCPClient:
    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        http: AsyncHttpClient | None = None,
        protocol_version: str = "2026-07-28",
    ) -> None:
        self.url = url
        self.headers = dict(headers or {})
        self.http = http or AsyncHttpClient()
        self._owns_http = http is None
        self.protocol_version = protocol_version
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}
        self._request_id = 0

    async def discover(self) -> dict[str, Any]:
        result = await self._request("server/discover", {})
        self.server_info = result.get("_meta", {}).get("io.modelcontextprotocol/serverInfo", {})
        self.capabilities = result.get("capabilities", {})
        return result

    async def initialize(self) -> dict[str, Any]:
        """Compatibility alias for callers migrating from the legacy API."""
        return await self.discover()

    async def close(self) -> None:
        if self._owns_http:
            await self.http.close()

    async def list_tools(self) -> list[RemoteTool]:
        definitions = await self._list_all("tools/list", "tools")
        return [RemoteTool(self, definition) for definition in definitions if _valid_tool_definition(definition)]

    async def list_tools_page(self, *, cursor: str | None = None, page_size: int | None = None) -> dict[str, Any]:
        return await self._list_page("tools/list", cursor=cursor, page_size=page_size)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, *, input_schema: dict[str, Any] | None = None) -> Any:
        params = {"name": name, "arguments": arguments or {}}
        if input_schema is not None:
            params["_schema"] = input_schema
        result = await self._request("tools/call", params)
        _ensure_complete_result(result)
        if result.get("isError"):
            raise MCPError(-32603, _content_text(result.get("content", [])), data=result)
        if result.get("resultType") == "input_required":
            raise MCPInputRequired(result.get("inputRequests", []), result.get("requestState"), data=result)
        if "structuredContent" in result:
            return result["structuredContent"]
        return _content_value(result.get("content", []))

    async def list_resources(self) -> dict[str, Any]:
        result = await self._list_all_result("resources/list", "resources")
        return result

    async def list_resources_page(self, *, cursor: str | None = None, page_size: int | None = None) -> dict[str, Any]:
        return await self._list_page("resources/list", cursor=cursor, page_size=page_size)

    async def list_resource_templates_page(self, *, cursor: str | None = None, page_size: int | None = None) -> dict[str, Any]:
        return await self._list_page("resources/templates/list", cursor=cursor, page_size=page_size)

    async def read_resource(self, uri: str) -> Any:
        result = await self._request("resources/read", {"uri": uri})
        _ensure_complete_result(result)
        return result.get("contents", [])

    async def list_prompts(self) -> dict[str, Any]:
        return await self._list_all_result("prompts/list", "prompts")

    async def list_prompts_page(self, *, cursor: str | None = None, page_size: int | None = None) -> dict[str, Any]:
        return await self._list_page("prompts/list", cursor=cursor, page_size=page_size)

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        result = await self._request("prompts/get", {"name": name, "arguments": arguments or {}})
        _ensure_complete_result(result)
        return result

    async def _list_all(self, method: str, key: str) -> list[dict[str, Any]]:
        result = await self._list_all_result(method, key)
        return result.get(key, [])

    async def _list_all_result(self, method: str, key: str) -> dict[str, Any]:
        result = await self._list_page(method)
        values = list(result.get(key, []))
        cursor = result.get("nextCursor")
        while cursor is not None:
            result = await self._list_page(method, cursor=cursor)
            values.extend(result.get(key, []))
            cursor = result.get("nextCursor")
        result[key] = values
        result.pop("nextCursor", None)
        return result

    async def _list_page(self, method: str, *, cursor: str | None = None, page_size: int | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        if page_size is not None:
            params["pageSize"] = page_size
        return await self._request(method, params)

    async def stream_events(self, method: str, params: dict[str, Any] | None = None):
        """Yield decoded SSE events from an MCP request."""
        self._request_id += 1
        data = self._payload(method, params, self._request_id)
        chunks = self.http.post_json_stream(
            self.url,
        headers=self._headers(method, params),
            data=data,
        )
        async for event in parse_sse_lines_async(chunks):
            yield event

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        try:
            response = await self.http.post_json(
                self.url, headers=self._headers(method, params),
                data=self._payload(method, params, self._request_id),
                raise_for_status=False,
            )
        except TypeError:  # compatibility with injected test transports
            response = await self.http.post_json(self.url, headers=self._headers(method, params), data=self._payload(method, params, self._request_id))
        if isinstance(response.body, str) and "text/event-stream" in response.headers.get("content-type", "").lower():
            events = [event async for event in parse_sse_lines_async(_single_chunk(response.body.encode()))]
            body = _find_jsonrpc_response(events, self._request_id)
        else:
            body = response.body
        _validate_jsonrpc_envelope(body, self._request_id)
        if body.get("error") is not None:
            error = body["error"]
            raise MCPError(error.get("code"), error.get("message", "MCP error"), data=error.get("data"), http_status=response.status_code)
        return body["result"]

    def _payload(self, method: str, params: dict[str, Any], request_id: int) -> dict[str, Any]:
        params = params or {}
        wire_params = {key: value for key, value in params.items() if key != "_schema"}
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": {
                **wire_params,
                "_meta": {
                    "io.modelcontextprotocol/protocolVersion": self.protocol_version,
                    "io.modelcontextprotocol/clientInfo": {"name": "modmex-ai", "version": "0.2.0"},
                    "io.modelcontextprotocol/clientCapabilities": {},
                    **wire_params.get("_meta", {}),
                },
            },
        }

    def _headers(self, method: str, params: dict[str, Any] | None = None) -> dict[str, str]:
        headers = {
            "accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self.protocol_version,
            "Mcp-Method": method,
            **self.headers,
        }
        if method == "tools/call" and params and isinstance(params.get("name"), str):
            headers["Mcp-Name"] = self._encode_header(params["name"])
        elif method == "prompts/get" and params and isinstance(params.get("name"), str):
            headers["Mcp-Name"] = self._encode_header(params["name"])
        elif method == "resources/read" and params and isinstance(params.get("uri"), str):
            # Resource URIs are not HTTP header tokens.  The transport's
            # base64 sentinel preserves them losslessly across intermediaries.
            headers["Mcp-Name"] = self._encode_header(params["uri"], force=True)
        schema = params.get("_schema") if params else None
        if isinstance(schema, dict) and isinstance(params.get("arguments"), dict):
            headers.update(extract_mcp_parameter_headers(input_schema=schema, arguments=params["arguments"]))
        return headers

    @staticmethod
    def _encode_header(value: str, *, force: bool = False) -> str:
        import base64
        if force or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~-" for char in value):
            return "=?base64?" + base64.b64encode(value.encode()).decode() + "?="
        return value


def _header(headers: dict[str, str], name: str) -> str | None:
    target = name.lower()
    return next((value for key, value in headers.items() if key.lower() == target), None)


def _content_text(content: list[dict[str, Any]]) -> str:
    for item in content:
        if item.get("type") == "text":
            return str(item.get("text", ""))
    return json.dumps(content, default=str)


def _content_value(content: list[dict[str, Any]]) -> Any:
    text = _content_text(content)
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


async def _single_chunk(data: bytes):
    yield data


def _ensure_complete_result(result: dict[str, Any]) -> None:
    result_type = result.get("resultType", "complete")
    if result_type == "input_required":
        raise MCPInputRequired(result.get("inputRequests", []), result.get("requestState"), data=result)
    if result_type != "complete":
        raise MCPError(-32603, f"Unsupported MCP resultType: {result_type}", data=result)


def _valid_tool_definition(definition: Any) -> bool:
    if not isinstance(definition, dict):
        return False
    schema = definition.get("inputSchema", {})
    if not isinstance(schema, dict):
        return False
    try:
        extract_mcp_parameter_headers(input_schema=schema, arguments={})
    except (TypeError, ValueError):
        return False
    return True


def _find_jsonrpc_response(messages: list[Any], request_id: int) -> dict[str, Any]:
    for message in messages:
        if isinstance(message, dict) and message.get("id") == request_id:
            return message
    raise MCPError(-32603, "MCP response did not contain the requested JSON-RPC id")


def _validate_jsonrpc_envelope(body: Any, request_id: int) -> None:
    if not isinstance(body, dict):
        raise MCPError(-32603, "MCP server returned a non-object JSON-RPC response")
    if body.get("jsonrpc") != "2.0":
        raise MCPError(-32603, "Invalid JSON-RPC version in MCP response", data=body)
    if body.get("id") != request_id:
        raise MCPError(-32603, "MCP response id does not match request id", data=body)
    has_result = "result" in body
    has_error = "error" in body
    if has_result == has_error:
        raise MCPError(-32603, "MCP response must contain exactly one of result or error", data=body)
