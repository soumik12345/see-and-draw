import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from inspect import isawaitable
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from tau_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentHarness,
    AgentHarnessConfig,
    AgentToolResult,
    EventListener,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tau_ai import ModelProvider

from see_and_draw.krita.client import KritaClient
from see_and_draw.krita.tools import create_krita_save_tool, create_krita_tools

KRITA_SYSTEM_PROMPT = """You are SeeAndDrawAgent, an autonomous digital-art agent controlling a local Krita session through Krita Codex MCP. Do not merely describe how to draw: inspect the live document, make deliberate native Krita edits, evaluate visible evidence, and iterate until the user's request is satisfied.

Operating principles

- Keep the workflow local-first and preserve the user's active document, filesystem boundaries, and undo history.
- Treat `krita_state` and the live canvas as authoritative. Never assume document dimensions, layer identifiers, brush state, compatibility, or save state.
- Reuse the exact `document_id` and `layer_id` values returned by tools. Never infer identifiers from layer names.
- Serialize all mutations. Never edit the same document concurrently.
- Prefer incremental, reversible edits. Before a destructive operation, identify the exact document, layer, region, and path affected.
- Treat a failed or timed-out mutation as having an unknown outcome. Never automatically retry it.
- Use only reported compatible capabilities, and never claim a visual change, native preset, save, or export succeeded without tool evidence.

Tool selection

- Use `krita_state` and `krita_preview` for inspection.
- Use `krita_document` for compatibility checks and document lifecycle operations.
- Use `krita_canvas` for geometry, composited color sampling, and raster-layer imports.
- Use `krita_layers` for stable-UUID layer creation, activation, properties, ordering, and transforms.
- Use `krita_brush`, `krita_paint_path`, and `krita_draw` for native preset-backed marks and geometry.
- Use `krita_edit` for selections, clipboard operations, fills, clears, and selection-to-layer operations.
- Use `krita_color` for project palettes, swatches, color replacement, and safe correction filters.
- Use `krita_history` for bounded undo and redo.
- Use `krita_asset` only for requested layer, animation, tileset, or named-region packages.
- Use `krita_save_export` only for a user-authorized checkpoint or export path. Leave the run's `artwork.kra` and `artwork.png` paths to the private runtime finalizer.

Brush selection

- Read or search installed presets with `krita_brush` instead of guessing names. Select a native preset that matches the requested medium, mark quality, scale, and painting stage.
- Establish explicit foreground, preset, size, opacity, flow, rotation, eraser mode, and pressure behavior when they matter. Do not rely on state left by the user or a prior batch.
- Use broad, simple native brushes for large masses, then progressively smaller or more controlled brushes for edges, texture, and details.
- Supply smooth ordered points to `krita_paint_path` and vary per-point pressure intentionally. Use `krita_draw` when geometry should be clean and regular.
- Inspect the rendered marks before changing presets or brush state. Make each change in response to visible edge, texture, coverage, or hierarchy problems.

Required workflow

1. Start every run with `krita_state`. Read connection, active document, canvas, layer tree, painting state, and `runtime_compatibility`. Stop rather than call an affected or unavailable feature.
2. If the bridge is unavailable, explain that the user must install and enable Krita Codex Bridge, restart Krita, and run `krita-codex-install check`. Do not install software, widen allowed roots, or change user-wide configuration without explicit permission.
3. Call `krita_preview` before changing visible content. Use a cropped preview when only a small region matters.
4. If an existing requested file is not open, use `krita_document` to open it. If new artwork has no usable document, create a moderate-size document with requested dimensions when specified, then call `krita_state` again and retain the returned document and layer UUIDs.
5. Make a concise visual plan covering composition, silhouettes, palette, value structure, layer order, and finishing criteria. Work in canvas coordinates and use separate named layers for major independently editable elements.
6. Build broad to fine: background and large value masses, secondary forms, contours, lighting, texture, then small accents. Prefer purposeful native shapes, selections, fills, and paths over dense noisy strokes.
7. After each meaningful visual change—or a small batch of related marks—call `krita_preview`. After document-lifecycle or layer-tree changes, call `krita_state` and refresh stored UUIDs.
8. Evaluate each preview for silhouette, proportion, overlap, perspective, value separation, color harmony, edge quality, focal hierarchy, and prompt fidelity. Choose the single highest-impact correction next.
9. If a change made the image worse, use `krita_history` for the smallest sufficient undo count, preview again, and try a materially different correction.
10. If a mutation times out, reports `operation_may_complete=true`, or returns `BRIDGE_BUSY`, do not retry it. Once the bridge responds, inspect with `krita_state` and `krita_preview` before deciding what remains.
11. Keep `overwrite=false` unless the user explicitly authorizes replacement of that exact target. Keep every path beneath configured allowed roots and never widen those roots for convenience.
12. Repeat the observe-plan-edit-review loop until the request is satisfied or further edits would not materially improve it. Reserve turns for final inspection.

Drawing judgment

- Start with readable shapes and strong value grouping. Detail cannot rescue an unclear silhouette or composition.
- Maintain spatial consistency across stroke batches. Reuse landmarks and canvas-relative proportions instead of guessing unrelated coordinates each turn.
- Separate overlapping subjects with value, color, contour, or negative space.
- Vary line weight and opacity deliberately; avoid uniform mechanical marks unless the requested style calls for them.
- For painterly work, layer larger low-detail masses before smaller high-contrast strokes. For graphic work, prefer stable shape primitives and clean boundaries.
- At thumbnail scale, the subject and focal point should remain recognizable. If they do not, simplify before adding detail.
- Do not imitate a living artist's distinctive style. Convert such requests into non-identifying visual characteristics such as palette, medium, mood, mark-making, and composition.

Saving and completion

- Before finishing, perform a final `krita_preview`. Correct obvious artifacts, accidental marks, clipped forms, weak contrast, or unfinished regions.
- Finish with a concise report of what changed and whether it was visually verified. Distinguish any user-requested KRA checkpoint from derived PNG or asset output and report only exact paths returned by tools.
- The runtime finalizer owns the supplied `artwork.kra` and `artwork.png` paths and persists `rollout.json` after normal completion. Do not call `krita_save_export` for those paths or claim they were saved yourself; the runtime's final artifact message is authoritative.
"""


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    run_id: str
    directory: Path
    kra_path: Path
    png_path: Path
    rollout_path: Path


def _image_input(image: ImageContent) -> dict[str, str]:
    return {
        "type": "input_image",
        "image_url": f"data:{image.mime_type};base64,{image.data}",
        "detail": "auto",
    }


def _user_message_input(message: UserMessage) -> dict[str, Any]:
    if isinstance(message.content, str):
        content: list[dict[str, str]] = [
            {"type": "input_text", "text": message.content}
        ]
    else:
        content = []
        for block in message.content:
            if isinstance(block, TextContent):
                content.append({"type": "input_text", "text": block.text})
            elif isinstance(block, ImageContent):
                content.append(_image_input(block))

    return {"type": "message", "role": "user", "content": content}


def _assistant_message_input(message: AssistantMessage) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    output_content: list[dict[str, Any]] = []

    def flush_output_content() -> None:
        if not output_content:
            return
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": list(output_content),
            }
        )
        output_content.clear()

    for block in message.content:
        if isinstance(block, TextContent):
            output_content.append(
                {"type": "output_text", "text": block.text, "annotations": []}
            )
        elif isinstance(block, ThinkingContent):
            output_content.append(
                {
                    "type": "output_text",
                    "text": block.thinking,
                    "annotations": [],
                }
            )
        elif isinstance(block, ToolCall):
            flush_output_content()
            items.append(
                {
                    "type": "function_call",
                    "call_id": block.id,
                    "name": block.name,
                    "arguments": json.dumps(
                        block.arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "status": "completed",
                }
            )

    if message.error_message:
        output_content.append(
            {
                "type": "output_text",
                "text": message.error_message,
                "annotations": [],
            }
        )

    flush_output_content()
    return items


def _tool_result_input(message: ToolResultMessage) -> dict[str, Any]:
    output: list[dict[str, str]] = []
    if message.is_error:
        output.append({"type": "input_text", "text": "Tool execution failed."})

    for block in message.content:
        if isinstance(block, TextContent):
            output.append({"type": "input_text", "text": block.text})
        elif isinstance(block, ImageContent):
            output.append(_image_input(block))

    if not output:
        output.append({"type": "input_text", "text": ""})

    return {
        "type": "function_call_output",
        "call_id": message.tool_call_id,
        "output": output,
    }


def build_openai_trajectory(
    messages: Sequence[AgentMessage],
    *,
    model: str,
) -> dict[str, Any]:
    """Convert one Tau run into a replayable OpenAI Responses API request."""
    input_items: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, UserMessage):
            input_items.append(_user_message_input(message))
        elif isinstance(message, AssistantMessage):
            input_items.extend(_assistant_message_input(message))
        elif isinstance(message, ToolResultMessage):
            input_items.append(_tool_result_input(message))

    return {
        "model": model,
        "instructions": KRITA_SYSTEM_PROMPT,
        "input": input_items,
    }


class KritaSeeAndDrawAgent:
    def __init__(
        self,
        provider: ModelProvider,
        model: str,
        *,
        max_turns: int = 200,
        krita_url: str = "http://127.0.0.1:5678",
        krita_command: str = "krita-codex-mcp",
        krita_args: Sequence[str] = (),
        runs_dir: str | Path = "runs",
    ) -> None:
        if krita_url != "http://127.0.0.1:5678":
            raise ValueError(
                "krita_url is no longer configurable; Krita Codex MCP owns its "
                "authenticated bridge configuration"
            )
        self.krita = KritaClient(command=krita_command, args=krita_args)
        self.runs_dir = Path(runs_dir)
        self.last_run_artifacts: RunArtifacts | None = None
        self.last_trajectory_path: Path | None = None
        self._listeners: list[EventListener] = []

        krita_tools = create_krita_tools(self.krita)
        self._save_tool = create_krita_save_tool(self.krita)
        self.harness = AgentHarness(
            AgentHarnessConfig(
                provider=provider,
                model=model,
                system=KRITA_SYSTEM_PROMPT,
                tools=krita_tools,
                max_turns=max_turns,
            )
        )

    async def run(self, prompt: str) -> AsyncIterator[AgentEvent]:
        message_start = len(self.harness.messages)
        run_id = uuid4().hex
        artifacts = self._create_run_artifacts(run_id)
        self.last_run_artifacts = artifacts
        self.last_trajectory_path = artifacts.rollout_path
        runtime_context = json.dumps(
            {
                "run_directory": str(artifacts.directory),
                "kra_path": str(artifacts.kra_path),
                "png_path": str(artifacts.png_path),
                "rollout_path": str(artifacts.rollout_path),
            },
            ensure_ascii=False,
        )
        run_prompt = (
            f"{prompt.rstrip()}\n\n"
            "Runtime context supplied by the caller:\n"
            f"{runtime_context}\n"
            "Complete and visually verify the artwork before the turn limit. "
            "The runtime finalizer will save the listed artifacts after normal "
            "completion."
        )

        terminal_event: AgentEndEvent | None = None
        try:
            async for event in self.harness.prompt(run_prompt):
                if isinstance(event, AgentEndEvent):
                    terminal_event = event
                else:
                    await self._notify(event)
                    yield event

            if terminal_event is not None and self._run_ended_normally(
                terminal_event.messages
            ):
                async for event in self._finalize_run(artifacts):
                    await self._notify(event)
                    yield event

            if terminal_event is not None:
                final_event = AgentEndEvent(
                    messages=list(self.harness.messages[message_start:])
                )
                await self._notify(final_event)
                yield final_event
        finally:
            messages = self.harness.messages[message_start:]
            self._export_trajectory(messages, artifacts.rollout_path)

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        """Subscribe to every event emitted by this Krita agent."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return unsubscribe

    async def _notify(self, event: AgentEvent) -> None:
        for listener in list(self._listeners):
            result = listener(event)
            if isawaitable(result):
                await result

    def _create_run_artifacts(self, run_id: str) -> RunArtifacts:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        existing_indexes = []
        for child in self.runs_dir.iterdir():
            index, separator, _ = child.name.partition("_")
            if child.is_dir() and separator and index.isdigit():
                existing_indexes.append(int(index))

        run_index = max(existing_indexes, default=0) + 1
        directory = (self.runs_dir / f"{run_index}_{run_id}").resolve()
        directory.mkdir()
        return RunArtifacts(
            run_id=run_id,
            directory=directory,
            kra_path=directory / "artwork.kra",
            png_path=directory / "artwork.png",
            rollout_path=directory / "rollout.json",
        )

    @staticmethod
    def _run_ended_normally(messages: Sequence[AgentMessage]) -> bool:
        final_assistant = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, AssistantMessage)
            ),
            None,
        )
        return final_assistant is not None and final_assistant.stop_reason == "stop"

    async def _finalize_run(
        self,
        artifacts: RunArtifacts,
    ) -> AsyncIterator[AgentEvent]:
        call = ToolCall(
            id=f"call-save-{uuid4().hex}",
            name="krita_save",
            arguments={"directory": str(artifacts.directory)},
        )
        call_message = AssistantMessage(
            model=self.harness.config.model,
            content=[call],
            stop_reason="toolUse",
        )
        self.harness.append_message(call_message)
        yield MessageStartEvent(message=call_message)
        yield MessageEndEvent(message=call_message)
        yield ToolExecutionStartEvent(
            tool_call_id=call.id,
            tool_name=call.name,
            args=dict(call.arguments),
        )

        is_error = False
        try:
            result = await self._save_tool.execute(call.id, call.arguments)
        except Exception as error:
            is_error = True
            result = AgentToolResult(
                content=[TextContent(text=f"Artifact saving failed: {error}")]
            )

        yield ToolExecutionEndEvent(
            tool_call_id=call.id,
            tool_name=call.name,
            result=result,
            is_error=is_error,
        )
        result_message = ToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            content=result.content,
            details=result.details,
            added_tool_names=result.added_tool_names,
            is_error=is_error,
        )
        self.harness.append_message(result_message)
        yield MessageStartEvent(message=result_message)
        yield MessageEndEvent(message=result_message)

        if is_error:
            final_text = (
                "The artwork run ended, but saving the KRA and PNG artifacts "
                f"failed.\n\nRollout: {artifacts.rollout_path}"
            )
        else:
            final_text = (
                "Artifacts saved:\n"
                f"- Editable KRA: {artifacts.kra_path}\n"
                f"- PNG preview: {artifacts.png_path}\n"
                f"- Rollout: {artifacts.rollout_path}"
            )
        final_message = AssistantMessage(
            model=self.harness.config.model,
            content=[TextContent(text=final_text)],
            stop_reason="stop",
        )
        self.harness.append_message(final_message)
        yield MessageStartEvent(message=final_message)
        yield MessageEndEvent(message=final_message)

    def _export_trajectory(
        self,
        messages: Sequence[AgentMessage],
        path: Path,
    ) -> Path:
        payload = build_openai_trajectory(
            messages,
            model=self.harness.config.model,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
        return path

    async def aclose(self) -> None:
        await self.krita.aclose()
