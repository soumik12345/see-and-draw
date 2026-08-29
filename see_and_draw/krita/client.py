from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from tau_agent import AgentToolResult
from tau_agent.messages import ImageContent, TextContent


class KritaClient:
    """Persistent MCP client for the local Krita Codex MCP server."""

    def __init__(
        self,
        command: str = "krita-codex-mcp",
        args: Sequence[str] = (),
    ) -> None:
        server_environment = {
            name: value
            for name in ("KRITA_CODEX_CONFIG", "LOCALAPPDATA")
            if (value := os.environ.get(name)) is not None
        }
        self._server_parameters = StdioServerParameters(
            command=command,
            args=list(args),
            env=server_environment or None,
        )
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._connect_lock = asyncio.Lock()
        self._call_lock = asyncio.Lock()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> AgentToolResult:
        session = await self._get_session()
        async with self._call_lock:
            result = await session.call_tool(name, arguments or {})

        content: list[TextContent | ImageContent] = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                content.append(TextContent(text=block.text))
            elif isinstance(block, types.ImageContent):
                mime_type = getattr(block, "mime_type", None) or getattr(
                    block, "mimeType"
                )
                content.append(ImageContent(data=block.data, mime_type=mime_type))

        structured = getattr(
            result,
            "structuredContent",
            getattr(result, "structured_content", None),
        )
        if not content and structured is not None:
            content.append(
                TextContent(
                    text=json.dumps(structured, ensure_ascii=False, sort_keys=True)
                )
            )
        if not content:
            content.append(TextContent(text="Krita tool completed successfully."))

        is_error = getattr(result, "isError", getattr(result, "is_error", False))
        if is_error:
            message = "\n".join(
                block.text for block in content if isinstance(block, TextContent)
            )
            raise RuntimeError(message or f"Krita tool {name} failed")

        if isinstance(structured, dict):
            details: dict[str, Any] = structured
        elif structured is None:
            details = {}
        else:
            details = {"structured_content": structured}
        return AgentToolResult(content=content, details=details)

    async def _get_session(self) -> ClientSession:
        if self._session is not None:
            return self._session

        async with self._connect_lock:
            if self._session is not None:
                return self._session

            stack = AsyncExitStack()
            try:
                read, write = await stack.enter_async_context(
                    stdio_client(self._server_parameters)
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
            except BaseException:
                await stack.aclose()
                raise

            self._stack = stack
            self._session = session
            return session

    async def aclose(self) -> None:
        stack = self._stack
        self._stack = None
        self._session = None
        if stack is not None:
            await stack.aclose()
