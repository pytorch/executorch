# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


Serializable = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


@dataclass
class ViewBlock:
    """Single renderable block for dashboard/record/compare views."""

    id: str
    title: str
    type: Literal["table", "html", "custom", "graph"]
    record: Dict[str, Any] = field(default_factory=dict)
    compare: Dict[str, Any] = field(default_factory=dict)
    order: int = 0
    collapsible: bool = True


@dataclass
class ViewList:
    """Ordered list of blocks returned by frontends."""

    blocks: List[ViewBlock] = field(default_factory=list)


@dataclass
class GraphView:
    """Convenience model for GraphView blocks."""

    id: str
    title: str
    graph_ref: str
    default_layers: List[str] = field(default_factory=list)
    default_color_by: Optional[str] = None
    layer_scope: Union[str, List[str]] = "all"
    viewer_options: Dict[str, Any] = field(default_factory=dict)
    controls: Dict[str, Any] = field(default_factory=dict)
    fullscreen: Dict[str, Any] = field(default_factory=dict)
    compare: Dict[str, Any] = field(
        default_factory=lambda: {
            "mode": "auto",
            "max_parallel": 2,
            "sync_toggle": True,
            "viewer_options_compare": {
                "layout_mode": "compare_compact",
                "sidebar_mode": "hidden",
                "minimap_mode": "off",
                "info_mode": "external",
            },
        }
    )
    order: int = 0
    collapsible: bool = True

    def as_block(self) -> ViewBlock:
        """Convert to canonical ViewBlock."""
        record = {
            "graph_ref": self.graph_ref,
            "default_layers": self.default_layers,
            "default_color_by": self.default_color_by,
            "layer_scope": self.layer_scope,
            "viewer_options": self.viewer_options,
            "controls": self.controls,
            "fullscreen": self.fullscreen,
        }
        return ViewBlock(
            id=self.id,
            title=self.title,
            type="graph",
            record=record,
            compare=self.compare,
            order=self.order,
            collapsible=self.collapsible,
        )


class Frontend:
    """Visualization strategy object returned by each lens."""

    def resources(self) -> Dict[str, str]:
        return {}

    def dashboard(
        self,
        start: Dict[str, Any],
        end: Dict[str, Any],
        analysis: Dict[str, Any],
        records: List[Any],
    ) -> Optional[ViewList]:
        return None

    def record(
        self,
        digest: Any,
        analysis: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[ViewList]:
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
    """Context shared across lens runtime hooks."""

    config: Dict[str, Any]
    shared_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordDigest:
    """Persistent observation item."""

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
    """Analysis output for dashboard and record rendering."""

    global_data: Dict[str, Serializable] = field(default_factory=dict)
    per_record_data: Dict[str, Serializable] = field(default_factory=dict)


class Lens:
    """Protocol for Observatory lenses."""

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

    @classmethod
    def contribute_graph_layers(
        cls,
        digest: Any,
        context: Dict[str, Any],
        graph_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return []

    @staticmethod
    def get_frontend_spec() -> Frontend:
        return Frontend()
