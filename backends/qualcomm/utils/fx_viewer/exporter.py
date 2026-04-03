from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict
from typing import Callable, List, Optional, Any, Dict

import networkx as nx
import torch
import torch.fx

from .color_rules import ColorRule
from .extension import GraphExtension
from .grandalf.layouts import SugiyamaLayout
from .grandalf.routing import route_with_lines
from .grandalf.utils.nx import convert_nextworkx_graph_to_grandalf
from .models import (
    BaseGraphPayload,
    GraphEdge,
    GraphExtensionPayload,
    GraphNode,
    GraphPayload,
)


class FXGraphExporter:
    """Export PyTorch FX graphs to JSON/JS/HTML payloads for the viewer.

    The exporter extracts node metadata from ``fx_node.meta`` into ``node.info``.
    Scalar meta values (str, int, float, bool) are included automatically.
    ``debug_handle`` is handled explicitly to support both scalar (int) and
    fused/tuple forms (tuple[int, ...] / list[int]):
      - int  → stored as int
      - tuple/list with one non-zero element → stored as int
      - tuple/list with multiple non-zero elements → stored as list[int]
    This ensures ``node.info.debug_handle`` is always present and usable by the
    JS compare sync engine (``mode: 'auto'`` set-intersection matching).
    """

    def __init__(self, graph_module: torch.fx.GraphModule):
        self.graph_module = graph_module
        self.extensions: List[GraphExtension] = []

        self.base_label_formatter: Callable[[GraphNode], str] = self._default_base_label
        self.base_tooltip_formatter: Callable[[GraphNode], List[str]] = self._default_base_tooltip
        self.base_color_rule: Optional[ColorRule] = None

    def _default_base_label(self, node: GraphNode) -> str:
        target = str(node.info.get("target") or node.info.get("op") or "")
        return target.replace("aten.", "").replace(".default", "")

    def _default_base_tooltip(self, node: GraphNode) -> List[str]:
        lines = [
            f"Name: {node.info.get('name', 'n/a')}",
            f"Op: {node.info.get('op', 'n/a')}",
            f"Target: {node.info.get('target', 'n/a')}",
        ]
        return lines

    def set_base_label_formatter(self, formatter: Callable[[GraphNode], str]):
        self.base_label_formatter = formatter

    def set_base_tooltip_formatter(self, formatter: Callable[[GraphNode], List[str]]):
        self.base_tooltip_formatter = formatter

    def set_base_color_rule(self, rule: ColorRule):
        self.base_color_rule = rule

    def add_extension(self, extension: GraphExtension):
        if not isinstance(extension, GraphExtension):
            raise TypeError("extension must be a GraphExtension")
        if any(ext.id == extension.id for ext in self.extensions):
            raise ValueError(f"duplicate extension id: '{extension.id}'")
        self.extensions.append(extension)

    @staticmethod
    def _get_from_node_root_name(from_node_list):
        """Walk from_node chain to root, return root node name."""
        if not from_node_list:
            return None
        ns = from_node_list[-1]
        while getattr(ns, "from_node", None):
            ns = ns.from_node[-1]
        return getattr(ns, "name", None)

    @staticmethod
    def _format_arg(arg):
        if isinstance(arg, torch.fx.Node):
            return arg.name
        if isinstance(arg, (list, tuple)):
            return type(arg)(FXGraphExporter._format_arg(a) for a in arg)
        if isinstance(arg, dict):
            return {k: FXGraphExporter._format_arg(v) for k, v in arg.items()}
        return str(arg)

    def _extract_graph(self) -> tuple[dict[str, GraphNode], list[GraphEdge]]:
        # torch.fx.Graph.nodes iterates in topological order (documented guarantee).
        print("Building graph payload model...")
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []

        for idx, fx_node in enumerate(self.graph_module.graph.nodes):
            info: dict[str, Any] = {
                "op": fx_node.op,
                "name": fx_node.name,
                "target": str(fx_node.target),
                "args": self._format_arg(fx_node.args),
                "kwargs": self._format_arg(fx_node.kwargs),
            }

            if "tensor_meta" in fx_node.meta:
                tm = fx_node.meta["tensor_meta"]
                if isinstance(tm, list):
                    info["tensor_shape"] = [tuple(t.shape) if hasattr(t, "shape") else None for t in tm]
                    info["dtype"] = [str(t.dtype) if hasattr(t, "dtype") else None for t in tm]
                elif hasattr(tm, "shape"):
                    info["tensor_shape"] = tuple(tm.shape)
                    info["dtype"] = str(tm.dtype) if hasattr(tm, "dtype") else None

            for key, value in fx_node.meta.items():
                if key != "tensor_meta" and isinstance(value, (str, int, float, bool)):
                    info[key] = value

            # Explicitly handle debug_handle (may be int or tuple — not caught by scalar loop)
            raw_dh = fx_node.meta.get("debug_handle")
            if raw_dh is not None and raw_dh != () and raw_dh != []:
                if isinstance(raw_dh, int):
                    info["debug_handle"] = raw_dh
                elif isinstance(raw_dh, (tuple, list)):
                    ints = [int(x) for x in raw_dh if isinstance(x, int) and x != 0]
                    if ints:
                        info["debug_handle"] = ints[0] if len(ints) == 1 else ints

            raw_fn = fx_node.meta.get("from_node")
            if raw_fn and isinstance(raw_fn, list) and len(raw_fn) > 0:
                root_name = self._get_from_node_root_name(raw_fn)
                if root_name:
                    info["from_node_root"] = root_name

            nodes[fx_node.name] = GraphNode(id=fx_node.name, topo_index=idx, info=info)

            for input_node in fx_node.all_input_nodes:
                edges.append(GraphEdge(v=input_node.name, w=fx_node.name))

        return nodes, edges

    @staticmethod
    def _validate_str_list(value: Any, *, context: str) -> list[str]:
        if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
            warnings.warn(f"{context} must return list[str]", RuntimeWarning, stacklevel=2)
            return []
        return value

    def _safe_base_label(self, node: GraphNode) -> str:
        label = self.base_label_formatter(node)
        if not isinstance(label, str):
            warnings.warn(
                f"base_label_formatter returned non-str for node '{node.id}', coercing to str",
                RuntimeWarning,
                stacklevel=2,
            )
            return str(label)
        return label

    def _safe_base_tooltip(self, node: GraphNode) -> list[str]:
        try:
            value = self.base_tooltip_formatter(node)
        except Exception as exc:
            warnings.warn(
                f"base_tooltip_formatter failed for node '{node.id}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return []
        return self._validate_str_list(value, context=f"base_tooltip_formatter(node='{node.id}')")

    def _ext_label_lines_for_layout(self, extension: GraphExtension, node_id: str) -> list[str]:
        if not extension.label_formatter or node_id not in extension.nodes_data:
            return []
        try:
            result = extension.label_formatter(extension.nodes_data[node_id])
        except Exception as exc:
            warnings.warn(
                f"Extension '{extension.id}' label formatter failed for node '{node_id}' during layout: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return []
        return self._validate_str_list(
            result,
            context=f"extension '{extension.id}' label formatter(node='{node_id}')",
        )

    def _compute_layout(self, nodes: dict[str, GraphNode], edges: list[GraphEdge]) -> None:
        print("Converting to Grandalf graph and computing layout...")
        graph_nx = nx.DiGraph()
        for node_id in nodes:
            graph_nx.add_node(node_id)
        for edge in edges:
            graph_nx.add_edge(edge.v, edge.w)

        g_grandalf = convert_nextworkx_graph_to_grandalf(graph_nx)

        edge_map = {(edge.v, edge.w): edge for edge in edges}

        class NodeView:
            def __init__(self, w, h):
                self.w = w
                self.h = h
                self.xy = (0, 0)

        class EdgeView:
            def __init__(self):
                self.points = []

            def setpath(self, points):
                self.points = points

        for vertex in g_grandalf.V():
            node = nodes[vertex.data]

            base_label = self._safe_base_label(node)
            max_char_width = len(base_label)
            total_lines = 1

            for ext in self.extensions:
                ext_lines = self._ext_label_lines_for_layout(ext, node.id)
                for line in ext_lines:
                    max_char_width = max(max_char_width, len(line))
                    total_lines += 1

            node.width = max(max_char_width * 7 + 20, 100)
            node.height = total_lines * 16 + 20
            vertex.view = NodeView(node.width, node.height)

        for edge in g_grandalf.E():
            edge.view = EdgeView()

        print("Running Sugiyama Layout...")
        for component in g_grandalf.C:
            sug = SugiyamaLayout(component)
            sug.route_edge = route_with_lines
            sug.xspace = 20
            sug.yspace = 40
            sug.init_all(optimize=True)
            sug.draw(N=5.5)

        for vertex in g_grandalf.V():
            node = nodes[vertex.data]
            node.x = float(vertex.view.xy[0])
            node.y = float(vertex.view.xy[1])

        for edge in g_grandalf.E():
            key = (edge.v[0].data, edge.v[1].data)
            if key not in edge_map:
                continue
            points = []
            if hasattr(edge, "view") and hasattr(edge.view, "points"):
                points = [{"x": float(p[0]), "y": float(p[1])} for p in edge.view.points]
            edge_map[key].points = points

    def _build_base_payload(self, nodes: dict[str, GraphNode], edges: list[GraphEdge]) -> BaseGraphPayload:
        print("[FX Graph Viewer] Compiling base graph payload...")

        base_color_input = {node_id: node.info for node_id, node in nodes.items()}
        base_colors: dict[str, str] = {}
        base_legend: list[dict[str, str]] = []
        if self.base_color_rule:
            base_colors, base_legend = self.base_color_rule.apply(base_color_input)

        for node in nodes.values():
            node.label = self._safe_base_label(node)
            node.tooltip = self._safe_base_tooltip(node)
            if node.id in base_colors:
                node.fill_color = base_colors[node.id]

        return BaseGraphPayload(
            legend=base_legend,
            nodes=list(nodes.values()),
            edges=edges,
        )

    def _build_extensions_payload(self) -> dict[str, GraphExtensionPayload]:
        print("[ FX Graph Viewer ] Compiling extension payloads...")
        return {ext.id: ext.build_payload() for ext in self.extensions}

    def generate_json_payload(self) -> Dict[str, Any]:
        nodes, edges = self._extract_graph()
        self._compute_layout(nodes, edges)
        base_payload = self._build_base_payload(nodes, edges)
        extensions_payload = self._build_extensions_payload()
        payload = GraphPayload(base=base_payload, extensions=extensions_payload)
        return asdict(payload)

    def export_json(self, output_path: str):
        data = self.generate_json_payload()
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Success! Exported JSON payload to {output_path}")

    @staticmethod
    def _load_viewer_js_bundle() -> str:
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        ordered_files = [
            "runtime.js",
            "graph_data_store.js",
            "search_engine.js",
            "view_controller.js",
            "canvas_renderer.js",
            "minimap_renderer.js",
            "ui_manager.js",
            "fx_graph_viewer.js",
            "compare.js",
        ]
        chunks = []
        for filename in ordered_files:
            path = os.path.join(template_dir, filename)
            with open(path, "r") as f:
                chunks.append(f"\n// ---- {filename} ----\n")
                chunks.append(f.read())
            
        return "\n".join(chunks)

    def export_js(self, container_id: str) -> str:
        data = self.generate_json_payload()
        json_str = json.dumps(data)
        js_content = self._load_viewer_js_bundle()

        return f"""
        const graphPayload = {json_str};

        {js_content}

        (function() {{
            try {{
                const viewer = FXGraphViewer.create({{
                    payload: graphPayload,
                    mount: {{ root: '#{container_id}' }},
                }});
                viewer.init();
                window.fxViewer = viewer;
            }} catch (e) {{
                console.error("Failed to mount FXGraphViewer:", e);
                const container = document.getElementById('{container_id}');
                if (container) {{
                    container.innerHTML = "<div style='color:red;'>Error mounting graph: " + e.message + "</div>";
                }}
            }}
        }})();
        """

    def export_html(self, output_html: str = "model_graph.html"):
        data = self.generate_json_payload()
        json_str = json.dumps(data)
        js_content = self._load_viewer_js_bundle()

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset=\"UTF-8\">
    <title>PyTorch FX Graph Viewer V3</title>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; font-family: sans-serif; background-color: #f5f5f5; }}
        #graph-viewer-container {{ width: 100%; height: 100%; }}
        #loading-overlay {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255, 255, 255, 0.95); display: flex; flex-direction: column; justify-content: center; align-items: center; z-index: 1000; }}
        .spinner {{ width: 40px; height: 40px; border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite; }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>
    <div id=\"loading-overlay\">
        <div class=\"spinner\"></div>
        <div id=\"loading-text\" style=\"margin-top: 20px; font-size: 18px; color: #333;\">Loading Graph Viewer...</div>
    </div>
    <div id=\"graph-viewer-container\"></div>

    <script>
        const graphPayload = {json_str};
    </script>
    <script>
        {js_content}
    </script>
    <script>
        window.onload = function() {{
            const overlay = document.getElementById('loading-overlay');
            try {{
                const viewer = FXGraphViewer.create({{
                    payload: graphPayload,
                    mount: {{ root: '#graph-viewer-container' }},
                }});
                viewer.init();
                overlay.style.display = 'none';
                window.fxViewer = viewer;
            }} catch (e) {{
                console.error(e);
                document.getElementById('loading-text').textContent = "Error: " + e.message;
            }}
        }};
    </script>
</body>
</html>
"""

        print(f"Writing to {output_html}...")
        with open(output_html, "w") as f:
            f.write(html_content)

        print(f"Success! Exported extensible graph to {output_html}")
