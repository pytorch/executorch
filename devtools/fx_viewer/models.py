from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphNode:
    """Wire-format node schema used by exporter and JSON payload."""

    id: str
    label: str = ""
    x: float = 0.0
    y: float = 0.0
    width: float = 100.0
    height: float = 40.0
    topo_index: int = 0
    info: dict[str, Any] = field(default_factory=dict)
    tooltip: list[str] = field(default_factory=list)
    fill_color: str | None = None


@dataclass
class GraphEdge:
    """Wire-format edge schema used by exporter and JSON payload."""

    v: str
    w: str
    points: list[dict[str, float]] = field(default_factory=list)


@dataclass
class BaseGraphPayload:
    """Nodes are in topological order (guaranteed by torch.fx.Graph.nodes iteration)."""

    legend: list[dict[str, str]]
    nodes: list[GraphNode]
    edges: list[GraphEdge]


@dataclass
class GraphExtensionNodePayload:
    """Wire-format extension node schema."""

    info: dict[str, Any] = field(default_factory=dict)
    tooltip: list[str] = field(default_factory=list)
    label_append: list[str] = field(default_factory=list)
    fill_color: str | None = None


@dataclass
class GraphExtensionPayload:
    """Wire-format extension layer schema."""

    id: str
    name: str
    legend: list[dict[str, str]] = field(default_factory=list)
    nodes: dict[str, GraphExtensionNodePayload] = field(default_factory=dict)
    sync_keys: list[str] = field(default_factory=list)


@dataclass
class GraphPayload:
    base: BaseGraphPayload
    extensions: dict[str, GraphExtensionPayload]
