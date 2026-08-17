"""High-level MCP connections usable by :class:`modmex_ai.Agent`."""

from __future__ import annotations

from modmex_ai.http.async_client import AsyncHttpClient
from modmex_ai.mcp.client import MCPClient
from modmex_ai.mcp.tools import RemoteTool
from modmex_ai.mcp.client import MCPHeaders


class MCPClientConnection:
    """Bind one MCP client to an agent's remote tools.

    Discovery is lazy and asynchronous.  The connection caches the tool
    catalog until :meth:`invalidate_tools_cache` is called.
    """

    def __init__(
        self,
        url_or_client: str | MCPClient | None = None,
        *,
        client: MCPClient | None = None,
        headers: MCPHeaders | None = None,
        http: AsyncHttpClient | None = None,
        discover: bool = True,
    ) -> None:
        if client is not None and url_or_client is not None:
            raise ValueError("Provide either url_or_client or client, not both")
        source = client if client is not None else url_or_client
        if isinstance(source, MCPClient):
            if headers is not None or http is not None:
                raise ValueError(
                    "headers and http require a URL, "
                    "not an injected MCPClient"
                )
            self.client = source
            self._owns_client = False
        elif isinstance(source, str):
            self.client = MCPClient(
                source,
                headers=headers,
                http=http,
            )
            self._owns_client = True
        else:
            raise TypeError("MCPClientConnection requires a URL or MCPClient")
        self.discover_on_connect = discover
        self._tools: list[RemoteTool] | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def tools(self) -> list[RemoteTool]:
        if self._tools is None:
            raise RuntimeError("MCP client is not connected; call await connect() first")
        return list(self._tools)

    async def connect(self) -> "MCPClientConnection":
        if self._connected and self._tools is not None:
            return self
        if self.discover_on_connect:
            await self.client.discover()
        self._tools = await self.client.list_tools()
        self._connected = True
        return self

    async def refresh_tools(self) -> list[RemoteTool]:
        if not self._connected:
            await self.connect()
        else:
            self._tools = await self.client.list_tools()
        return self.tools

    def invalidate_tools_cache(self) -> None:
        self._tools = None
        self._connected = False

    async def close(self) -> None:
        self._connected = False
        self._tools = None
        if self._owns_client:
            await self.client.close()

    async def __aenter__(self) -> "MCPClientConnection":
        return await self.connect()

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()


class MCPClientManager:
    """Connect and clean up a group of MCP client connections."""

    def __init__(self, connections: list[MCPClientConnection]) -> None:
        self.connections = list(connections)
        self.active_connections: list[MCPClientConnection] = []

    async def connect(self) -> list[MCPClientConnection]:
        connected: list[MCPClientConnection] = []
        try:
            for connection in self.connections:
                await connection.connect()
                connected.append(connection)
        except Exception:
            await self._close_connections(reversed(connected))
            raise
        self.active_connections = connected
        return list(self.active_connections)

    @property
    def clients(self) -> list[MCPClientConnection]:
        return list(self.active_connections)

    async def close(self) -> None:
        await self._close_connections(reversed(self.active_connections or self.connections))
        self.active_connections = []

    async def _close_connections(self, connections) -> None:
        for connection in connections:
            await connection.close()

    async def __aenter__(self) -> list[MCPClientConnection]:
        return await self.connect()

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()
