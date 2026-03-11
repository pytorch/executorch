from __future__ import annotations

from typing import Dict, Any, Callable, Optional
import warnings

from .color_rules import ColorRule


class GraphExtension:
    """Optional annotation layer attached to the base FX graph."""

    def __init__(self, id: str, name: str):
        clean_id = id.strip()
        if not clean_id:
            raise ValueError("GraphExtension id must be non-empty")
        if not name.strip():
            raise ValueError("GraphExtension name must be non-empty")

        self.id = clean_id
        self.name = name
        self.nodes_data: Dict[str, Dict[str, Any]] = {}

        self.color_rule: Optional[ColorRule] = None
        self.label_formatter: Optional[Callable[[Dict[str, Any]], list[str]]] = None
        self.tooltip_formatter: Optional[Callable[[Dict[str, Any]], list[str]]] = None

    def add_node_data(self, node_id: str, data: Dict[str, Any]):
        if node_id not in self.nodes_data:
            self.nodes_data[node_id] = {}
        self.nodes_data[node_id].update(data)

    def set_color_rule(self, rule: ColorRule):
        self.color_rule = rule

    def set_label_formatter(self, formatter: Callable[[Dict[str, Any]], list[str]]):
        self.label_formatter = formatter

    def set_tooltip_formatter(self, formatter: Callable[[Dict[str, Any]], list[str]]):
        self.tooltip_formatter = formatter

    def _format_lines(
        self,
        *,
        formatter: Callable[[Dict[str, Any]], list[str]],
        data: Dict[str, Any],
        node_id: str,
        kind: str,
    ) -> list[str]:
        try:
            result = formatter(data)
        except Exception as exc:
            warnings.warn(
                f"Extension '{self.id}' {kind} formatter failed for node '{node_id}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return []

        if not isinstance(result, list) or any(not isinstance(x, str) for x in result):
            warnings.warn(
                f"Extension '{self.id}' {kind} formatter must return list[str] for node '{node_id}'",
                RuntimeWarning,
                stacklevel=2,
            )
            return []

        return result

    def build(self) -> Dict[str, Any]:
        node_colors = {}
        legend = []

        if self.color_rule:
            node_colors, legend = self.color_rule.apply(self.nodes_data)

        compiled_nodes = {}

        for node_id, data in self.nodes_data.items():
            compiled = {"info": data}

            if self.label_formatter:
                lines = self._format_lines(
                    formatter=self.label_formatter,
                    data=data,
                    node_id=node_id,
                    kind="label",
                )
                if lines:
                    compiled["label_append"] = lines

            if self.tooltip_formatter:
                lines = self._format_lines(
                    formatter=self.tooltip_formatter,
                    data=data,
                    node_id=node_id,
                    kind="tooltip",
                )
                if lines:
                    compiled["tooltip"] = lines

            if node_id in node_colors:
                compiled["fill_color"] = node_colors[node_id]

            compiled_nodes[node_id] = compiled

        return {
            "name": self.name,
            "legend": legend,
            "nodes": compiled_nodes,
        }
