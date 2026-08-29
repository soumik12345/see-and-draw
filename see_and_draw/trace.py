from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text
from tau_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnStartEvent,
)
from tau_agent.messages import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    UserMessage,
)


class RichTraceRenderer:
    """Render a compact, streaming Tau agent trace without binary payloads."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        show_thinking: bool = True,
        max_result_chars: int = 2_000,
    ) -> None:
        self.console = console or Console()
        self.show_thinking = show_thinking
        self.max_result_chars = max_result_chars
        self._turn = 0
        self._streamed_text = False
        self._streamed_thinking = False
        self._open_stream: str | None = None

    def __call__(self, event: AgentEvent) -> None:
        if isinstance(event, AgentStartEvent):
            self.console.rule("[bold cyan]See & Draw")
        elif isinstance(event, TurnStartEvent):
            self._turn += 1
            self.console.print(f"[dim]• turn {self._turn}[/dim]")
        elif isinstance(event, MessageStartEvent):
            self._handle_message_start(event)
        elif isinstance(event, MessageUpdateEvent):
            self._handle_message_update(event)
        elif isinstance(event, MessageEndEvent):
            self._handle_message_end(event)
        elif isinstance(event, ToolExecutionStartEvent):
            self._handle_tool_start(event)
        elif isinstance(event, ToolExecutionUpdateEvent):
            self._handle_tool_update(event)
        elif isinstance(event, ToolExecutionEndEvent):
            self._handle_tool_end(event)
        elif isinstance(event, AgentEndEvent):
            self._close_stream()
            self.console.rule("[bold green]Run complete")

    def _handle_message_start(self, event: MessageStartEvent) -> None:
        if isinstance(event.message, AssistantMessage):
            self._streamed_text = False
            self._streamed_thinking = False
            self._close_stream()

    def _handle_message_update(self, event: MessageUpdateEvent) -> None:
        update = event.assistant_message_event
        event_type = update.type
        delta = getattr(update, "delta", "")

        if event_type == "thinking_delta" and self.show_thinking:
            self._start_stream("thinking")
            self._streamed_thinking = True
            self.console.print(Text(delta, style="dim italic"), end="")
        elif event_type == "text_delta":
            self._start_stream("assistant")
            self._streamed_text = True
            self.console.print(Text(delta), end="")
        elif event_type == "thinking_end" and self._open_stream == "thinking":
            self._close_stream()
        elif event_type == "text_end" and self._open_stream == "assistant":
            self._close_stream()

    def _handle_message_end(self, event: MessageEndEvent) -> None:
        self._close_stream()
        message = event.message
        if isinstance(message, UserMessage):
            prompt = self._user_text(message)
            if prompt:
                self.console.print(
                    Panel(prompt, title="[bold]prompt", border_style="blue")
                )
            return
        if not isinstance(message, AssistantMessage):
            return

        if self.show_thinking and not self._streamed_thinking:
            thinking = "\n".join(
                block.thinking
                for block in message.content
                if isinstance(block, ThinkingContent)
            )
            if thinking:
                self.console.print(
                    Panel(
                        Text(thinking, style="dim italic"),
                        title="thinking",
                        border_style="bright_black",
                    )
                )

        if not self._streamed_text:
            text = "\n".join(
                block.text
                for block in message.content
                if isinstance(block, TextContent)
            )
            if text:
                self.console.print(
                    Panel(
                        Markdown(text),
                        title="[bold green]assistant",
                        border_style="green",
                    )
                )

    def _handle_tool_start(self, event: ToolExecutionStartEvent) -> None:
        self.console.print(
            Panel(
                Pretty(
                    self._sanitize(event.args),
                    max_length=20,
                    max_string=300,
                    max_depth=5,
                ),
                title=f"[bold cyan]tool[/bold cyan] {event.tool_name}",
                border_style="cyan",
            )
        )

    def _handle_tool_update(self, event: ToolExecutionUpdateEvent) -> None:
        summary = self._result_text(event.partial_result.content)
        if summary:
            self.console.print(f"[dim]↳ {event.tool_name}: {summary}[/dim]")

    def _handle_tool_end(self, event: ToolExecutionEndEvent) -> None:
        summary = self._result_text(event.result.content) or "No textual output"
        renderables: list[Any] = [Text(summary)]
        if event.result.details is not None:
            renderables.append(
                Pretty(
                    self._sanitize(event.result.details),
                    max_length=20,
                    max_string=300,
                    max_depth=5,
                )
            )

        status = "failed" if event.is_error else "done"
        color = "red" if event.is_error else "green"
        self.console.print(
            Panel(
                Group(*renderables),
                title=f"[bold {color}]{status}[/bold {color}] {event.tool_name}",
                border_style=color,
            )
        )

    def _start_stream(self, stream: str) -> None:
        if self._open_stream == stream:
            return
        self._close_stream()
        label = "thinking" if stream == "thinking" else "assistant"
        style = "dim italic" if stream == "thinking" else "bold green"
        self.console.print(Text(f"{label} › ", style=style), end="")
        self._open_stream = stream

    def _close_stream(self) -> None:
        if self._open_stream is not None:
            self.console.print()
            self._open_stream = None

    def _user_text(self, message: UserMessage) -> str:
        if isinstance(message.content, str):
            text = message.content
        else:
            text = "\n".join(
                block.text
                for block in message.content
                if isinstance(block, TextContent)
            )
        return text.partition("\n\nRuntime context supplied by the caller:")[0]

    def _result_text(self, content: list[TextContent | ImageContent]) -> str:
        parts = []
        for block in content:
            if isinstance(block, TextContent):
                parts.append(self._truncate(block.text))
            elif isinstance(block, ImageContent):
                parts.append(
                    f"<{block.mime_type} image omitted; "
                    f"{len(block.data)} base64 characters>"
                )
        return "\n".join(parts)

    def _truncate(self, text: str) -> str:
        prefix, separator, _ = text.partition("base64,")
        if separator:
            text = f"{prefix}base64,<omitted>"
        if len(text) <= self.max_result_chars:
            return text
        omitted = len(text) - self.max_result_chars
        return f"{text[: self.max_result_chars]}… <{omitted} characters omitted>"

    def _sanitize(self, value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {
                item_key: self._sanitize(item_value, str(item_key))
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [self._sanitize(item, key) for item in value]
        if isinstance(value, str):
            normalized_key = key.lower()
            if value.startswith("data:") or normalized_key in {
                "data",
                "image",
                "image_url",
            }:
                return f"<{len(value)} characters omitted>"
            return self._truncate(value)
        return value
