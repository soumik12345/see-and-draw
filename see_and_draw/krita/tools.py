from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tau_agent import AgentTool, AgentToolResult
from tau_agent.messages import TextContent

from see_and_draw.krita.client import KritaClient

JsonSchema = dict[str, Any]


def _string(*, enum: list[str] | None = None) -> JsonSchema:
    schema: JsonSchema = {"type": "string"}
    if enum is not None:
        schema["enum"] = enum
    return schema


def _integer(*, minimum: int | None = None, maximum: int | None = None) -> JsonSchema:
    schema: JsonSchema = {"type": "integer"}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def _number(
    *, minimum: float | None = None, maximum: float | None = None
) -> JsonSchema:
    schema: JsonSchema = {"type": "number"}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def _array(
    items: JsonSchema, *, minimum: int | None = None, maximum: int | None = None
) -> JsonSchema:
    schema: JsonSchema = {"type": "array", "items": items}
    if minimum is not None:
        schema["minItems"] = minimum
    if maximum is not None:
        schema["maxItems"] = maximum
    return schema


def _object(
    properties: dict[str, JsonSchema],
    *,
    required: list[str] | None = None,
) -> JsonSchema:
    schema: JsonSchema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


BOOLEAN: JsonSchema = {"type": "boolean"}
OBJECT: JsonSchema = {"type": "object"}
POINT = _object(
    {
        "x": _number(minimum=-1_000_000, maximum=1_000_000),
        "y": _number(minimum=-1_000_000, maximum=1_000_000),
        "pressure": _number(minimum=0, maximum=1),
    },
    required=["x", "y"],
)


@dataclass(frozen=True, slots=True)
class _ToolSpec:
    name: str
    label: str
    description: str
    parameters: JsonSchema


TOOL_SPECS = (
    _ToolSpec(
        "krita_state",
        "Inspect Krita state",
        "Read the connection, active document, recursive layer tree, painting state, and runtime compatibility report.",
        _object({"document_id": _string()}),
    ),
    _ToolSpec(
        "krita_preview",
        "Preview the Krita canvas",
        "Return a whole-canvas or pixel-precise cropped PNG preview as image content.",
        _object(
            {
                "max_width": _integer(minimum=64, maximum=4096),
                "max_height": _integer(minimum=64, maximum=4096),
                "document_id": _string(),
                "x": _integer(minimum=0, maximum=1_000_000),
                "y": _integer(minimum=0, maximum=1_000_000),
                "width": _integer(minimum=1, maximum=32768),
                "height": _integer(minimum=1, maximum=32768),
                "scale": _integer(minimum=1, maximum=16),
            }
        ),
    ),
    _ToolSpec(
        "krita_canvas",
        "Manage the Krita canvas",
        "Inspect or sample the canvas; resize, scale, crop, rotate, flip, or import an editable raster layer.",
        _object(
            {
                "action": _string(
                    enum=[
                        "read",
                        "sample_color",
                        "resize",
                        "scale",
                        "crop",
                        "rotate",
                        "flip",
                        "import_layer",
                    ]
                ),
                "document_id": _string(),
                "x": _integer(minimum=-1_000_000, maximum=1_000_000),
                "y": _integer(minimum=-1_000_000, maximum=1_000_000),
                "width": _integer(minimum=1, maximum=32768),
                "height": _integer(minimum=1, maximum=32768),
                "anchor": _string(
                    enum=[
                        "top_left",
                        "top",
                        "top_right",
                        "left",
                        "center",
                        "right",
                        "bottom_left",
                        "bottom",
                        "bottom_right",
                    ]
                ),
                "strategy": _string(),
                "angle_degrees": _number(minimum=-360, maximum=360),
                "direction": _string(enum=["horizontal", "vertical"]),
                "path": _string(),
                "name": _string(),
                "parent_id": _string(),
                "above_id": _string(),
            }
        ),
    ),
    _ToolSpec(
        "krita_edit",
        "Edit Krita selections",
        "Create or refine selections and perform native copy, cut, paste, fill, clear, and selection-to-layer operations.",
        _object(
            {
                "action": _string(
                    enum=[
                        "selection_status",
                        "select_rect",
                        "select_all",
                        "select_opaque",
                        "deselect",
                        "invert",
                        "grow",
                        "shrink",
                        "move_selection",
                        "resize_selection",
                        "copy",
                        "cut",
                        "paste",
                        "copy_to_layer",
                        "cut_to_layer",
                        "fill",
                        "clear",
                    ]
                ),
                "document_id": _string(),
                "layer_id": _string(),
                "x": _integer(minimum=-1_000_000, maximum=1_000_000),
                "y": _integer(minimum=-1_000_000, maximum=1_000_000),
                "width": _integer(minimum=1, maximum=32768),
                "height": _integer(minimum=1, maximum=32768),
                "combine": _string(enum=["replace", "add", "subtract", "intersect"]),
                "amount": _integer(minimum=1, maximum=4096),
                "color": _string(),
            }
        ),
    ),
    _ToolSpec(
        "krita_document",
        "Manage Krita documents",
        "Inspect compatibility and list, create, open, activate, save, or close Krita documents.",
        _object(
            {
                "action": _string(
                    enum=[
                        "list",
                        "compatibility",
                        "create",
                        "open",
                        "activate",
                        "save",
                        "close",
                    ]
                ),
                "path": _string(),
                "document_id": _string(),
                "width": _integer(minimum=1, maximum=32768),
                "height": _integer(minimum=1, maximum=32768),
                "name": _string(),
                "color_model": _string(),
                "color_depth": _string(),
                "profile": _string(),
                "resolution": _number(minimum=1, maximum=2400),
                "discard_changes": BOOLEAN,
            }
        ),
    ),
    _ToolSpec(
        "krita_layers",
        "Manage Krita layers",
        "Read and edit the layer tree by stable UUID, including properties, ordering, movement, transforms, and merge-down.",
        _object(
            {
                "action": _string(
                    enum=[
                        "read",
                        "create",
                        "activate",
                        "rename",
                        "set_properties",
                        "duplicate",
                        "delete",
                        "reorder",
                        "translate",
                        "scale",
                        "rotate",
                        "shear",
                        "crop",
                        "flip",
                        "merge_down",
                    ]
                ),
                "document_id": _string(),
                "layer_id": _string(),
                "name": _string(),
                "kind": _string(enum=["paintlayer", "grouplayer"]),
                "parent_id": _string(),
                "above_id": _string(),
                "visible": BOOLEAN,
                "opacity": _number(minimum=0, maximum=1),
                "locked": BOOLEAN,
                "alpha_locked": BOOLEAN,
                "inherit_alpha": BOOLEAN,
                "blending_mode": _string(),
                "offset_x": _integer(minimum=-1_000_000, maximum=1_000_000),
                "offset_y": _integer(minimum=-1_000_000, maximum=1_000_000),
                "x": _integer(minimum=-1_000_000, maximum=1_000_000),
                "y": _integer(minimum=-1_000_000, maximum=1_000_000),
                "width": _integer(minimum=1, maximum=32768),
                "height": _integer(minimum=1, maximum=32768),
                "strategy": _string(),
                "angle_degrees": _number(minimum=-360, maximum=360),
                "shear_x_degrees": _number(minimum=-89, maximum=89),
                "shear_y_degrees": _number(minimum=-89, maximum=89),
                "direction": _string(enum=["horizontal", "vertical"]),
                "transform_mode": _string(enum=["mask", "direct"]),
                "response_mode": _string(enum=["compact", "full"]),
                "checkpoint_reopen": BOOLEAN,
            }
        ),
    ),
    _ToolSpec(
        "krita_history",
        "Use Krita history",
        "Read undo and redo availability or execute a bounded number of native undo or redo steps.",
        _object(
            {
                "action": _string(enum=["status", "undo", "redo"]),
                "steps": _integer(minimum=1, maximum=100),
                "document_id": _string(),
            }
        ),
    ),
    _ToolSpec(
        "krita_brush",
        "Manage the native Krita brush",
        "Read or set native brush state and search installed Krita presets.",
        _object(
            {
                "action": _string(enum=["read", "list_presets", "set"]),
                "document_id": _string(),
                "query": _string(),
                "limit": _integer(minimum=1, maximum=500),
                "preset_name": _string(),
                "brush_size": _number(minimum=0.000001, maximum=1000),
                "opacity": _number(minimum=0, maximum=1),
                "flow": _number(minimum=0, maximum=1),
                "rotation": _number(minimum=-3600, maximum=3600),
                "eraser_mode": BOOLEAN,
                "disable_pressure": BOOLEAN,
                "foreground": _string(),
            }
        ),
    ),
    _ToolSpec(
        "krita_paint_path",
        "Paint a native Krita path",
        "Paint 2 to 512 pressure-aware line points with the current or named native brush preset.",
        _object(
            {
                "points": _array(POINT, minimum=2, maximum=512),
                "document_id": _string(),
                "layer_id": _string(),
                "preset_name": _string(),
                "brush_size": _number(minimum=0.000001, maximum=1000),
            },
            required=["points"],
        ),
    ),
    _ToolSpec(
        "krita_draw",
        "Draw native Krita geometry",
        "Draw native preset-backed lines, paths, polygons, rectangles, or ellipses with explicit stroke and fill styles.",
        _object(
            {
                "shape": _string(
                    enum=["line", "path", "rectangle", "ellipse", "polygon"]
                ),
                "document_id": _string(),
                "layer_id": _string(),
                "points": _array(POINT),
                "x": _number(minimum=-1_000_000, maximum=1_000_000),
                "y": _number(minimum=-1_000_000, maximum=1_000_000),
                "width": _number(minimum=0.000001, maximum=32768),
                "height": _number(minimum=0.000001, maximum=32768),
                "closed": BOOLEAN,
                "stroke_style": _string(
                    enum=["None", "ForegroundColor", "BackgroundColor"]
                ),
                "fill_style": _string(
                    enum=["None", "ForegroundColor", "BackgroundColor"]
                ),
                "preset_name": _string(),
                "brush_size": _number(minimum=0.000001, maximum=1000),
                "foreground": _string(),
                "background": _string(),
            },
            required=["shape"],
        ),
    ),
    _ToolSpec(
        "krita_color",
        "Manage Krita colors",
        "Use project colors and Krita palettes, replace colors, or apply an allow-listed correction filter.",
        _object(
            {
                "action": _string(
                    enum=[
                        "read",
                        "list_palettes",
                        "read_palette",
                        "set_project_palette",
                        "apply_swatch",
                        "replace_color",
                        "adjust",
                    ]
                ),
                "document_id": _string(),
                "layer_id": _string(),
                "palette_name": _string(),
                "swatch_index": _integer(minimum=0, maximum=4999),
                "target": _string(enum=["foreground", "background"]),
                "project_name": _string(),
                "entries": _array(OBJECT),
                "from_color": _string(),
                "to_color": _string(),
                "tolerance": _integer(minimum=0, maximum=255),
                "filter_name": _string(
                    enum=[
                        "autocontrast",
                        "brightnesscontrast",
                        "desaturate",
                        "hsvadjustment",
                        "invert",
                        "levels",
                        "normalize",
                    ]
                ),
                "settings": OBJECT,
                "x": _integer(minimum=-1_000_000, maximum=1_000_000),
                "y": _integer(minimum=-1_000_000, maximum=1_000_000),
                "width": _integer(minimum=1, maximum=32768),
                "height": _integer(minimum=1, maximum=32768),
                "limit": _integer(minimum=1, maximum=500),
            }
        ),
    ),
    _ToolSpec(
        "krita_asset",
        "Export Krita assets",
        "Export a layer, animation, tileset, or named-region package with a direct PNG preview.",
        _object(
            {
                "layer_id": _string(),
                "output_path": _string(),
                "action": _string(
                    enum=[
                        "export_layer",
                        "export_animation",
                        "export_tileset",
                        "export_regions",
                    ]
                ),
                "crop": _string(enum=["canvas", "content", "union"]),
                "overwrite": BOOLEAN,
                "document_id": _string(),
                "preview_max_width": _integer(minimum=64, maximum=4096),
                "preview_max_height": _integer(minimum=64, maximum=4096),
                "frame_ids": _array(_string()),
                "animation_name": _string(),
                "columns": _integer(minimum=1, maximum=64),
                "padding": _integer(minimum=0, maximum=64),
                "frame_duration_ms": _integer(minimum=1, maximum=60000),
                "pivot_x": _number(minimum=0, maximum=1),
                "pivot_y": _number(minimum=0, maximum=1),
                "pixels_per_unit": _number(minimum=0.000001, maximum=10000),
                "loop": BOOLEAN,
                "regions": _array(OBJECT),
                "package_name": _string(),
                "tile_width": _integer(minimum=1, maximum=32768),
                "tile_height": _integer(minimum=1, maximum=32768),
                "origin_x": _integer(minimum=0, maximum=1_000_000),
                "origin_y": _integer(minimum=0, maximum=1_000_000),
                "grid_columns": _integer(minimum=1, maximum=1024),
                "grid_rows": _integer(minimum=1, maximum=1024),
                "spacing_x": _integer(minimum=0, maximum=32768),
                "spacing_y": _integer(minimum=0, maximum=32768),
                "skip_empty": BOOLEAN,
                "name_prefix": _string(),
            },
            required=["layer_id", "output_path"],
        ),
    ),
    _ToolSpec(
        "krita_save_export",
        "Save and export Krita artwork",
        "Save a KRA checkpoint and/or export PNG beneath the configured allowed roots.",
        _object(
            {
                "kra_path": _string(),
                "png_path": _string(),
                "overwrite": BOOLEAN,
                "document_id": _string(),
            }
        ),
    ),
)


def _create_tool(client: KritaClient, spec: _ToolSpec) -> AgentTool:
    async def execute(tool_call_id, arguments, signal=None, on_update=None):
        return await client.call_tool(spec.name, arguments)

    return AgentTool(
        name=spec.name,
        label=spec.label,
        description=spec.description,
        parameters=spec.parameters,
        execute_fn=execute,
        execution_mode="sequential",
    )


def create_krita_tools(client: KritaClient) -> list[AgentTool]:
    """Create the complete Krita Codex MCP tool surface for Tau."""
    return [_create_tool(client, spec) for spec in TOOL_SPECS]


def create_krita_save_tool(client: KritaClient) -> AgentTool:
    """Create the runtime-only dual-format artifact finalizer."""

    async def execute(tool_call_id, arguments, signal=None, on_update=None):
        directory = Path(arguments["directory"]).resolve()
        kra_path = directory / "artwork.kra"
        png_path = directory / "artwork.png"
        await client.call_tool(
            "krita_save_export",
            {
                "kra_path": str(kra_path),
                "png_path": str(png_path),
                "overwrite": False,
            },
        )

        for path in (kra_path, png_path):
            if not path.exists():
                raise RuntimeError(f"Krita did not create {path}")

        return AgentToolResult(
            content=[
                TextContent(
                    text=f"Editable artwork: {kra_path}\nPNG preview: {png_path}"
                )
            ],
            details={
                "kra_path": str(kra_path),
                "png_path": str(png_path),
            },
        )

    return AgentTool(
        name="krita_save",
        label="Save run artifacts",
        description="Save the run as artwork.kra and artwork.png.",
        parameters=_object({"directory": _string()}, required=["directory"]),
        execute_fn=execute,
        execution_mode="sequential",
    )
