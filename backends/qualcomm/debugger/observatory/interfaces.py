# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed API contracts for the Observatory runtime.

This module is the source-of-truth contract for:
1. Frontend view composition (`ViewList` + typed blocks/specs).
2. Runtime record/session objects (`ObservationContext`, `RecordDigest`, etc.).
3. Analyze-phase graph layer contribution via fx_viewer payload types.

Architecture model:
1. Runtime phase (`observe`/`digest`) captures raw record data.
2. Analyze phase (`analyze`) computes global and per-record derived data.
3. Frontend phase (`Frontend.*`) maps typed data into renderable view blocks.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

if TYPE_CHECKING:
    from executorch.backends.qualcomm.utils.fx_viewer.extension import GraphExtension
    from executorch.backends.qualcomm.utils.fx_viewer.models import GraphExtensionPayload


# Type Alias for JSON-serializable leaf/object values.
Serializable = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


# ---------------------------------------------------------------------------
# Frontend block contracts
# ---------------------------------------------------------------------------


@dataclass
class TableRecordSpec:
    """Record payload for a table block.

    Args:
        data: Key-value pairs rendered in the default table renderer.
    """

    data: Dict[str, Serializable] = field(default_factory=dict)


@dataclass
class TableCompareSpec:
    """Compare behavior for table blocks.

    Modes:
    1. `auto`: runtime renders side-by-side table diff view.
    2. `disabled`: hide compare section for this block.
    """

    mode: Literal["auto", "disabled"] = "auto"


@dataclass
class HtmlRecordSpec:
    """Record payload for an HTML block.

    Args:
        content: Raw HTML fragment rendered into block content container.
    """

    content: str = ""


@dataclass
class HtmlCompareSpec:
    """Compare behavior for HTML blocks.

    Modes:
    1. `auto`: runtime renders selected HTML blocks side-by-side.
    2. `disabled`: hide compare section for this block.
    """

    mode: Literal["auto", "disabled"] = "auto"


@dataclass
class CustomRecordSpec:
    """Record payload for a custom JS block.

    JS signature:
      function renderRecord(container, args, context, analysis)

    JS argument mapping:
    1. `container`: host DOM container created by observatory runtime.
    2. `args`: exactly this dataclass field (`CustomRecordSpec.args`).
    3. `context`: runtime-selected context object.
       - Record view path: `{ index, record }`
         - `index`: record index in report payload.
         - `record`: full serialized record object from report payload.
       - Dashboard path: `{ start, end, records }`
         - `start`/`end`: lens session payloads.
         - `records`: full serialized records list.
    4. `analysis`: `report.analysis_results[lens_name]` object:
       `{ global_data, per_record_data }`.

    Practical access pattern for record callbacks:
    1. Digest: `context.record.digests[lens_name]`.
    2. Per-record analysis:
       `analysis.per_record_data?.[context.record.name]?.data`.
    3. Global analysis: `analysis.global_data`.

    Fields:
    1. `js_func`: global function path.
    2. `args`: static serializable args.
    """

    js_func: str = ""
    args: Dict[str, Serializable] = field(default_factory=dict)


@dataclass
class CustomCompareSpec:
    """Compare behavior for custom JS blocks.

    JS signature:
      function renderCompare(container, args, context, analysis)

    JS argument mapping:
    1. `container`: compare section DOM container.
    2. `args`: exactly this dataclass field (`CustomCompareSpec.args`).
    3. `context`:
       - `indices`: selected global record indices.
       - `names`: selected record names.
       - `records`: selected serialized record objects.
       - `blocks`: selected block payloads for this block ID.
       - `lens`: lens name.
       - `block_id`: block ID.
    4. `analysis`: `report.analysis_results[lens_name]` object:
       `{ global_data, per_record_data }`.

    Fields:
    1. `mode`: `custom` or `disabled`.
    2. `js_func`: compare function path. If omitted in `custom` mode, runtime
       falls back to `record.js_func`.
    3. `args`: static compare args.
    """

    mode: Literal["custom", "disabled"] = "disabled"
    js_func: Optional[str] = None
    args: Dict[str, Serializable] = field(default_factory=dict)


GraphLayerScope = Union[Literal["all", "lens_only"], List[str]]


@dataclass
class GraphRecordSpec:
    """Record payload for GraphView blocks.

    Core fields:
    1. `graph_ref`: key into report `graph_assets` and `graph_layers`.
    2. `default_layers`: initial extension layer IDs.
    3. `default_color_by`: initial color-by layer ID.
    4. `layer_scope`: `all`, `lens_only`, or explicit allowlist.
    5. `viewer_options`: passthrough options for embedded FX viewer.
    """

    graph_ref: str
    default_layers: List[str] = field(default_factory=list)
    default_color_by: Optional[str] = None
    layer_scope: GraphLayerScope = "all"
    viewer_options: Dict[str, Serializable] = field(default_factory=dict)
    controls: Dict[str, Serializable] = field(default_factory=dict)
    fullscreen: Dict[str, Serializable] = field(default_factory=dict)


@dataclass
class GraphCompareSpec:
    """Compare behavior for graph blocks.

    Modes:
    1. `auto`: runtime mounts side-by-side graph compare with optional sync.
    2. `custom`: call user JS function for compare rendering.
    3. `disabled`: hide compare section.
    """

    mode: Literal["auto", "disabled", "custom"] = "auto"
    max_parallel: int = 2
    sync_toggle: bool = True
    viewer_options_compare: Dict[str, Serializable] = field(
        default_factory=lambda: {
            "layout_mode": "compare_compact",
            "sidebar_mode": "hidden",
            "minimap_mode": "off",
            "info_mode": "external",
        }
    )
    js_func: Optional[str] = None
    args: Dict[str, Serializable] = field(default_factory=dict)


@dataclass
class TableBlock:
    """Typed view block for key-value table rendering."""

    id: str
    title: str
    record: TableRecordSpec
    compare: TableCompareSpec = field(default_factory=TableCompareSpec)
    order: int = 0
    collapsible: bool = True
    type: Literal["table"] = "table"


@dataclass
class HtmlBlock:
    """Typed view block for raw HTML rendering."""

    id: str
    title: str
    record: HtmlRecordSpec
    compare: HtmlCompareSpec = field(default_factory=HtmlCompareSpec)
    order: int = 0
    collapsible: bool = True
    type: Literal["html"] = "html"


@dataclass
class CustomBlock:
    """Typed view block for custom JS rendering."""

    id: str
    title: str
    record: CustomRecordSpec
    compare: CustomCompareSpec = field(default_factory=CustomCompareSpec)
    order: int = 0
    collapsible: bool = True
    type: Literal["custom"] = "custom"


@dataclass
class GraphBlock:
    """Typed view block for graph viewer rendering."""

    id: str
    title: str
    record: GraphRecordSpec
    compare: GraphCompareSpec = field(default_factory=GraphCompareSpec)
    order: int = 0
    collapsible: bool = True
    type: Literal["graph"] = "graph"


ViewBlock = Union[TableBlock, HtmlBlock, CustomBlock, GraphBlock]


@dataclass
class ViewList:
    """Ordered block list returned by lens frontends.

    Rules:
    1. Block IDs must be unique within one ViewList.
    2. Rendering order is controlled by `block.order`.
    """

    blocks: List[ViewBlock] = field(default_factory=list)


@dataclass
class GraphView:
    """Convenience authoring helper for one graph block.

    This helper is intended for lens authors who want the ergonomics of a
    focused graph API while still returning canonical `GraphBlock`.
    """

    id: str
    title: str
    graph_ref: str
    default_layers: List[str] = field(default_factory=list)
    default_color_by: Optional[str] = None
    layer_scope: GraphLayerScope = "all"
    viewer_options: Dict[str, Serializable] = field(default_factory=dict)
    controls: Dict[str, Serializable] = field(default_factory=dict)
    fullscreen: Dict[str, Serializable] = field(default_factory=dict)
    compare: GraphCompareSpec = field(default_factory=GraphCompareSpec)
    order: int = 0
    collapsible: bool = True

    def as_block(self) -> GraphBlock:
        """Build canonical `GraphBlock` from convenience fields."""

        return GraphBlock(
            id=self.id,
            title=self.title,
            record=GraphRecordSpec(
                graph_ref=self.graph_ref,
                default_layers=self.default_layers,
                default_color_by=self.default_color_by,
                layer_scope=self.layer_scope,
                viewer_options=self.viewer_options,
                controls=self.controls,
                fullscreen=self.fullscreen,
            ),
            compare=self.compare,
            order=self.order,
            collapsible=self.collapsible,
        )


def _require_non_empty_text(value: str, field_name: str, block_id: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ViewBlock '{block_id}' requires non-empty {field_name}")


def _require_str_list(value: Any, field_name: str, block_id: str) -> None:
    if not isinstance(value, list) or any((not isinstance(x, str) or not x.strip()) for x in value):
        raise ValueError(f"ViewBlock '{block_id}' requires {field_name} as list[str]")


def validate_view_block(block: ViewBlock) -> None:
    """Validate one typed frontend block.

    Validation covers:
    1. Required identity fields (`id`, `title`).
    2. Per-block invariant checks.
    3. Compare-mode specific requirements (for custom/graph blocks).
    """

    if not isinstance(block, (TableBlock, HtmlBlock, CustomBlock, GraphBlock)):
        raise TypeError(f"Unsupported ViewBlock type: {type(block)}")

    _require_non_empty_text(block.id, "id", "<unknown>")
    _require_non_empty_text(block.title, "title", block.id)

    if not isinstance(block.order, int):
        raise TypeError(f"ViewBlock '{block.id}' order must be int")

    if isinstance(block, CustomBlock):
        _require_non_empty_text(block.record.js_func, "record.js_func", block.id)
        if block.compare.mode == "custom":
            compare_js_func = (block.compare.js_func or "").strip() or block.record.js_func.strip()
            if not compare_js_func:
                raise ValueError(f"CustomBlock '{block.id}' compare mode custom requires js_func")

    if isinstance(block, GraphBlock):
        _require_non_empty_text(block.record.graph_ref, "record.graph_ref", block.id)

        if block.record.default_layers:
            _require_str_list(block.record.default_layers, "record.default_layers", block.id)

        scope = block.record.layer_scope
        if isinstance(scope, str):
            if scope not in {"all", "lens_only"}:
                raise ValueError(
                    f"GraphBlock '{block.id}' layer_scope must be 'all', 'lens_only', or list[str]"
                )
        elif isinstance(scope, list):
            _require_str_list(scope, "record.layer_scope", block.id)
        else:
            raise ValueError(
                f"GraphBlock '{block.id}' layer_scope must be 'all', 'lens_only', or list[str]"
            )

        if int(block.compare.max_parallel) < 1:
            raise ValueError(f"GraphBlock '{block.id}' compare.max_parallel must be >= 1")

        if block.compare.mode == "custom" and not (block.compare.js_func or "").strip():
            raise ValueError(f"GraphBlock '{block.id}' compare mode custom requires js_func")


def validate_view_list(view_list: ViewList) -> None:
    """Validate a full frontend `ViewList` contract."""

    if not isinstance(view_list, ViewList):
        raise TypeError(f"Expected ViewList, got {type(view_list)}")

    seen_ids = set()
    for block in view_list.blocks:
        validate_view_block(block)
        if block.id in seen_ids:
            raise ValueError(f"Duplicate ViewBlock id in one ViewList: {block.id}")
        seen_ids.add(block.id)


# ---------------------------------------------------------------------------
# Analysis contracts
# ---------------------------------------------------------------------------


@dataclass
class GraphLayerContribution:
    """Graph layer contribution attached during analyze-phase.

    `extension` accepts either:
    1. `GraphExtensionPayload` (preferred stable payload type).
    2. `GraphExtension` (authoring helper converted lazily).
    """

    extension: Union["GraphExtension", "GraphExtensionPayload"]
    id_override: Optional[str] = None
    name_override: Optional[str] = None

    def to_payload(self) -> "GraphExtensionPayload":
        """Resolve contribution into a `GraphExtensionPayload`."""

        from executorch.backends.qualcomm.utils.fx_viewer.extension import GraphExtension
        from executorch.backends.qualcomm.utils.fx_viewer.models import GraphExtensionPayload

        payload: GraphExtensionPayload
        if isinstance(self.extension, GraphExtensionPayload):
            payload = self.extension
        elif isinstance(self.extension, GraphExtension):
            payload = self.extension.build_payload()
        else:
            raise TypeError(
                "GraphLayerContribution.extension must be GraphExtensionPayload or GraphExtension"
            )

        if self.id_override or self.name_override:
            return GraphExtensionPayload(
                id=self.id_override or payload.id,
                name=self.name_override or payload.name,
                legend=payload.legend,
                nodes=payload.nodes,
            )

        return payload


@dataclass
class RecordAnalysis:
    """Per-record analysis output.

    Fields:
    1. `data`: record-specific derived values consumed by frontend record views.
    2. `graph_layers`: map from local layer key to typed graph contribution.
    """

    data: Dict[str, Serializable] = field(default_factory=dict)
    graph_layers: Dict[str, GraphLayerContribution] = field(default_factory=dict)

    def add_graph_layer(
        self,
        key: str,
        extension: Union["GraphExtension", "GraphExtensionPayload"],
        *,
        id_override: Optional[str] = None,
        name_override: Optional[str] = None,
    ) -> None:
        """Add or replace a graph layer contribution for this record."""

        if not key.strip():
            raise ValueError("RecordAnalysis graph layer key must be non-empty")
        self.graph_layers[key] = GraphLayerContribution(
            extension=extension,
            id_override=id_override,
            name_override=name_override,
        )


# ---------------------------------------------------------------------------
# Runtime core contracts
# ---------------------------------------------------------------------------


class Frontend:
    """Visualization strategy object returned by each lens.

    Frontend methods are block-oriented:
    1. `dashboard(...) -> ViewList | None`
    2. `record(...) -> ViewList | None`

    Compare behavior is declared per block (`block.compare`) instead of a
    separate lens-level `compare()` callback.
    """

    def resources(self) -> Dict[str, str]:
        """Return optional shared JS/CSS resources.

        Returns:
            Dict with optional keys:
            1. `js`: inline JavaScript source.
            2. `css`: inline CSS source.
        """

        return {}

    def dashboard(
        self,
        start: Dict[str, Any],
        end: Dict[str, Any],
        analysis: Dict[str, Any],
        records: List[Any],
    ) -> Optional[ViewList]:
        """Build dashboard-level block list for one lens.

        Python-side inputs:
        1. `start` <- `SessionResult.start_data[lens_name]`.
        2. `end` <- `SessionResult.end_data[lens_name]`.
        3. `analysis` <- `AnalysisResult.global_data` (this lens).
        4. `records` <- collected `RecordDigest` list.

        Render dataflow:
        1. Return `ViewList(blocks=[...])`.
        2. Blocks are serialized into report payload.
        3. For `CustomBlock`, JS callback receives:
           `fn(container, block.record.args, {start,end,records}, analysis_results[lens_name])`.

        Args:
            start: Session start payload from `on_session_start`.
            end: Session end payload from `on_session_end`.
            analysis: Lens global analysis payload.
            records: Serialized record list for context-aware summaries.
        """
        return None

    def record(
        self,
        digest: Any,
        analysis: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[ViewList]:
        """Build record-level block list for one lens.

        Python-side inputs:
        1. `digest` <- current record digest for this lens.
        2. `analysis` <- `{ "global": global_data, "record": per_record_data[name].data }`.
        3. `context` <- `{ "index": int, "name": str }`.

        Render dataflow:
        1. Return `ViewList(blocks=[...])`.
        2. Runtime mounts block renderers per selected record.
        3. For `CustomBlock`, JS callback receives:
           `fn(container, block.record.args, {index, record}, analysis_results[lens_name])`.
        4. JS `record` is the serialized report record object, so digest data is
           available via `context.record.digests[lens_name]`.

        Args:
            digest: Current record digest for this lens.
            analysis: Dict with `global` and `record` derived analysis.
            context: Record context metadata (`index`, `name`).
        """
        return None

    def check_badges(self, digest: Any, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        return []

    def check_index_diffs(
        self,
        prev_digest: Any,
        curr_digest: Any,
        analysis: Dict[str, Any],
    ) -> Dict[str, str]:
        return {}


@dataclass
class ObservationContext:
    """Context shared across runtime lens hooks.

    `shared_state` is a per-collect broker for cross-lens hints (for example,
    exposing record name or artifact hints discovered by one lens).
    """

    config: Dict[str, Any]
    shared_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordDigest:
    """Persistent observation item.

    This is the canonical persisted unit produced by runtime capture.
    """

    name: str
    timestamp: float
    data: Dict[str, Serializable] = field(default_factory=dict)


@dataclass
class SessionResult:
    """Session start/end data from lens hooks."""

    start_data: Dict[str, Serializable] = field(default_factory=dict)
    end_data: Dict[str, Serializable] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Global + per-record analysis contract for a lens."""

    global_data: Dict[str, Serializable] = field(default_factory=dict)
    per_record_data: Dict[str, RecordAnalysis] = field(default_factory=dict)


class Lens:
    """Protocol for Observatory lenses.

    Lifecycle phases:
    1. Runtime (stateful): `setup`, session hooks, `observe`, `digest`, `clear`.
    2. Analyze (pure-data): `analyze(records, config)`.
    3. Frontend strategy: `get_frontend_spec()`.
    """

    @classmethod
    def get_name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def setup(cls) -> None:
        pass

    @classmethod
    def on_session_start(cls, context: ObservationContext) -> Optional[Serializable]:
        return None

    @classmethod
    def observe(cls, artifact: Any, context: ObservationContext) -> Any:
        return None

    @classmethod
    def digest(cls, observation: Any, context: ObservationContext) -> Serializable:
        return None

    @classmethod
    def on_session_end(cls, context: ObservationContext) -> Optional[Serializable]:
        return None

    @classmethod
    def clear(cls) -> None:
        pass

    @staticmethod
    def analyze(records: List[RecordDigest], config: Dict[str, Any]) -> AnalysisResult:
        return AnalysisResult()

    @staticmethod
    def get_frontend_spec() -> Frontend:
        return Frontend()
