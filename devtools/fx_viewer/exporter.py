from __future__ import annotations

import copy
import json
import os
import warnings
from dataclasses import asdict
from typing import Callable, List, Optional, Any, Dict, Sequence

import torch
import torch.fx

from .color_rules import ColorRule
from .extension import GraphExtension

try:
    from executorch.exir.dialects.edge._ops import EdgeOpOverload
except ImportError:
    EdgeOpOverload = None
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

    _NODE_CHAR_WIDTH = 7
    _NODE_MIN_WIDTH = 100
    _NODE_X_PADDING = 20
    _NODE_LINE_HEIGHT = 16
    _NODE_Y_PADDING = 20
    _LAYOUT_XSPACE = 50
    _LAYOUT_YSPACE = 30
    _DUMMY_SIZE_X = 100  # dummy nodes (from fast-sugiyama) occupy no real width/height
    _DUMMY_SIZE_Y = 30  # dummy nodes (from fast-sugiyama) occupy no real width/height
    _SPINE_COHESION_ITER = 20

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
            target = fx_node.target
            schema = None
            if EdgeOpOverload is not None and isinstance(target, EdgeOpOverload):
                target_str = target.__name__
                schema = target._schema.schema
            elif isinstance(target, torch._ops.OpOverload):
                target_str = str(target)
                schema = getattr(target, "_schema", None)
            else:
                target_str = str(target)

            info: dict[str, Any] = {
                "op": fx_node.op,
                "name": fx_node.name,
                "target": target_str,
                "args": self._format_arg(fx_node.args),
                "kwargs": self._format_arg(fx_node.kwargs),
            }

            if schema is not None:
                info["schema"] = str(schema)
                pos_schema_args = [
                    a for a in schema.arguments if not a.kwarg_only
                ]
                formatted_args = self._format_arg(fx_node.args)
                if not isinstance(formatted_args, (list, tuple)):
                    formatted_args = (formatted_args,) if formatted_args else ()
                named = {}
                for i, val in enumerate(formatted_args):
                    name = (
                        pos_schema_args[i].name
                        if i < len(pos_schema_args)
                        else f"arg_{i}"
                    )
                    named[name] = val
                formatted_kwargs = self._format_arg(fx_node.kwargs)
                if isinstance(formatted_kwargs, dict):
                    named.update(formatted_kwargs)
                info["named_args"] = named

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

    @classmethod
    def _compute_node_box_size(
        cls,
        base_label: str,
        extension_lines: Sequence[str] | None = None,
    ) -> tuple[int, int]:
        max_char_width = len(base_label or "")
        total_lines = 1

        if extension_lines:
            for line in extension_lines:
                max_char_width = max(max_char_width, len(line))
                total_lines += 1

        width = max(max_char_width * cls._NODE_CHAR_WIDTH + cls._NODE_X_PADDING, cls._NODE_MIN_WIDTH)
        height = total_lines * cls._NODE_LINE_HEIGHT + cls._NODE_Y_PADDING
        return width, height

    @classmethod
    def _compute_layout_with_ext_lines(
        cls,
        nodes: dict[str, GraphNode],
        edges: list[GraphEdge],
        ext_label_lines_by_node: dict[str, list[str]],
        base_label_getter: Callable[[GraphNode], str],
    ) -> None:
        for node_id, node in nodes.items():
            base_label = base_label_getter(node)
            ext_lines = ext_label_lines_by_node.get(node.id, [])
            node.width, node.height = cls._compute_node_box_size(base_label, ext_lines)

        if not nodes:
            return

        try:
            from fast_sugiyama import from_edges
        except ImportError as exc:
            raise ImportError(
                "fx_viewer layout requires 'fast-sugiyama' (and rectangle-packer "
                "for multi-component packing). Install with: "
                "pip install 'fast-sugiyama[all]'  (requires Python >= 3.11)"
            ) from exc

        print("Running fast-sugiyama layout...")
        edge_list = [(e.v, e.w) for e in edges if e.v in nodes and e.w in nodes]
        widths = [n.width for n in nodes.values()]
        # Median width as vertex_spacing keeps the layout tight for typical
        # nodes; wide outliers are handled by the per-layer compaction below.
        from statistics import median
        baseline_w = median(widths)
        max_w = max(widths)
        expected_gap = baseline_w + cls._LAYOUT_XSPACE

        raw_layouts = from_edges(
            edge_list,
            vertex_spacing=expected_gap,
            minimum_length=1,
            dummy_vertices=True,
            crossing_minimization="median",
        )

        # Adaptive per-layer x-spacing + y-row compaction using actual node
        # sizes. Preserves fast-sugiyama's within-layer ordering (so crossings
        # are preserved) while tightening gaps for narrow nodes and widening
        # them for long labels.
        adjusted = cls._compact_components(raw_layouts, nodes, expected_gap)

        # Isolated nodes (zero edges in the edge_list) are invisible to
        # from_edges; synthesize a one-node component for each so rect_pack
        # tiles them into the layout instead of leaving them stacked at 0,0.
        referenced: set = set()
        for a, b in edge_list:
            referenced.add(a)
            referenced.add(b)
        for nid, node in nodes.items():
            if nid not in referenced:
                adjusted.append(
                    ([(nid, (0.0, 0.0))], float(node.width), float(node.height), [])
                )

        # Pack components. fast-sugiyama's bbox is center-to-center only, so
        # pad spacing by max_w + xspace to prevent edge-level overlap between
        # adjacent components.
        from fast_sugiyama.layout import Layouts
        pack_spacing = max_w + cls._LAYOUT_XSPACE
        widest_component = max((w for _pos, w, _h, _e in adjusted), default=0.0)
        pack_width = int(max(widest_component*3 + pack_spacing, 2000.0))
        layouts = Layouts(adjusted).rect_pack_layouts(
            max_width=pack_width,
            spacing=pack_spacing,
        )

        positions: dict[Any, tuple[float, float]] = {}
        expanded_edges: list[tuple[Any, Any]] = []
        for positions_list, _w, _h, edges_with_dummies in layouts:
            positions.update(dict(positions_list))
            if edges_with_dummies:
                expanded_edges.extend(edges_with_dummies)

        for node_id, node in nodes.items():
            if node_id in positions:
                x, y = positions[node_id]
                node.x = float(x)
                node.y = float(y)

        edge_map = {(e.v, e.w): e for e in edges}
        for (u, v), pts in cls._polylines_from_dummy_chain(
            expanded_edges, nodes, positions
        ).items():
            if (u, v) not in edge_map:
                continue
            clipped = cls._clip_edge_polyline(pts, nodes[u], nodes[v])
            edge_map[(u, v)].points = [
                {"x": float(x), "y": float(y)} for (x, y) in clipped
            ]

    @classmethod
    def _compact_components(cls, layouts, nodes, expected_gap):
        from collections import defaultdict

        def _w(nid):
            n = nodes.get(nid)
            return n.width if n is not None else cls._DUMMY_SIZE_X

        def _h(nid):
            n = nodes.get(nid)
            return n.height if n is not None else cls._DUMMY_SIZE_Y

        xspace = cls._LAYOUT_XSPACE
        yspace = cls._LAYOUT_YSPACE
        expected = max(float(expected_gap), 1.0)

        new_layouts = []
        for pos, w, h, el in layouts:
            x_orig: dict = {nid: px for nid, (px, _) in pos}
            x: dict = dict(x_orig)

            by_y: dict = defaultdict(list)
            for nid, (_, py) in pos:
                by_y[py].append(nid)

            def _sweep_min_gap(nids):
                nids.sort(key=lambda n: x[n])
                for i in range(1, len(nids)):
                    prev, cur = nids[i - 1], nids[i]
                    min_gap = (_w(prev) + _w(cur)) / 2 + xspace
                    if x[cur] - x[prev] < min_gap:
                        x[cur] = x[prev] + min_gap

            # Phase 1: chain detection (real + dummy members)
            chains = cls._detect_chains(el or [], nodes)

            FAIR_RUNS=5
            # Phase 2: iterative spine cohesion + pure-A overlap fix
            for i in range(cls._SPINE_COHESION_ITER + FAIR_RUNS):
                
                # delta for each node
                # We record the relative weight (chain length) and their base delta
                node_delta: dict = defaultdict(list)
                for ch in chains:
                    if not ch:
                        continue
                    mean_x = sum(x[v] for v in ch) / len(ch)
                    # emphasize end point (disable for last FAIR_RUNS iters)
                    if i < cls._SPINE_COHESION_ITER:
                        # we use common start and end node of chain to attract chain close together
                        mean_x = (mean_x + x[ch[0]] + x[ch[-1]]) / 3.0
                    for v in ch:
                        # weight: len(ch)
                        # delta: mean_x - x[v]
                        node_delta[v].append((len(ch), (mean_x - x[v])))
                # move the node x
                for n, deltas in node_delta.items():
                    total_weight = sum(w for w, _ in deltas)
                    x[n] += sum(w / total_weight * d for w, d in deltas)
                # adjust node overlapping
                for nids in by_y.values():
                    _sweep_min_gap(nids)


            # Phase 3: vertical compaction with flipped y so inputs land
            # at the top of the canvas and outputs at the bottom. The
            # iteration runs largest-original-y first so that rank
            # receives new_y = 0 (top of canvas) and deeper ranks get
            # monotonically larger new_y values.
            distinct_ys = sorted(by_y.keys(), reverse=True)
            layer_h = {
                y: max((_h(n) for n in by_y[y]), default=0.0)
                for y in distinct_ys
            }
            new_y: dict = {}
            cursor = 0.0
            for i, y in enumerate(distinct_ys):
                if i == 0:
                    new_y[y] = cursor
                else:
                    cursor += (
                        layer_h[distinct_ys[i - 1]] + layer_h[y]
                    ) / 2 + yspace
                    new_y[y] = cursor

            new_positions = [(nid, (x[nid], new_y[py])) for nid, (_, py) in pos]
            xs = [xy[0] for _, xy in new_positions]
            ys = [xy[1] for _, xy in new_positions]
            new_w = (max(xs) - min(xs)) if xs else w
            new_h = (max(ys) - min(ys)) if ys else h
            new_layouts.append((new_positions, new_w, new_h, el))
        return new_layouts

    @classmethod
    def _detect_chains(cls, edge_list, nodes):
        # Here we break the graph into chains (connected node list)
        # The longer the chain the better (aligned visual vertical axis)
        # We achieve long chain by calculating best_prev and best_succ with rank
        # We must let the chain start and end node to be shared
        # The shared end points (common nodes) will be used to pull chains near in later iterative loop
        from collections import defaultdict

        if not edge_list:
            return []

        succ: dict = defaultdict(set)
        prev: dict = defaultdict(set)
        for u, v in edge_list:
            succ[u].add(v)
            prev[v].add(u)

        all_nodes = set(succ) | set(prev)

        # max depth a node's output can reach
        node_out_rank: dict = defaultdict(int)
        # the succer node that have maximal node_out_rank
        best_succ: dict = {}
        graph_output_nodes = [n for n in all_nodes if len(succ[n]) == 0]
        stack = graph_output_nodes
        while stack:
            n = stack.pop()
            for pn in prev[n]:
                score = 2 if pn in nodes else 1
                if node_out_rank[pn] < node_out_rank[n] + score:
                    node_out_rank[pn] = node_out_rank[n] + score
                    stack.append(pn)
                    best_succ[pn] = n

        # max depth a node's input can reach
        node_in_rank: dict = defaultdict(int)
        # the prev node that have maximal node_in_rank
        best_prev: dict = {}
        graph_input_nodes = [n for n in all_nodes if len(prev[n]) == 0]
        stack = graph_input_nodes
        while stack:
            n = stack.pop()
            for nn in succ[n]:
                score = 2 if nn in nodes else 1
                if node_in_rank[nn] < node_in_rank[n] + score:
                    node_in_rank[nn] = node_in_rank[n] + score
                    stack.append(nn)
                    best_prev[nn] = n


        visited: set = set()
        chains: list = []
        for start in sorted(all_nodes, key=lambda n:node_out_rank[n], reverse=True):
            if start in visited:
                continue
            cur = start
            walk = [start]
            if start in best_prev:
                walk.insert(0, best_prev[start])
            while cur not in visited:
                visited.add(cur)
                if cur not in best_succ:
                    break
                nxt = best_succ[cur]
                walk.append(nxt)
                cur = nxt
            if len(walk) >= 2: # always true
                chains.append(walk)
        return chains

    @staticmethod
    def _polylines_from_dummy_chain(
        expanded_edges: list[tuple[Any, Any]],
        nodes: dict[str, GraphNode],
        positions: dict[Any, tuple[float, float]],
    ) -> dict[tuple[Any, Any], list[tuple[float, float]]]:
        forward: dict[Any, Any] = {}
        for u, v in expanded_edges:
            forward[u] = v

        polylines: dict[tuple[Any, Any], list[tuple[float, float]]] = {}
        for u, v in expanded_edges:
            if u not in nodes:
                continue
            chain: list[tuple[float, float]] = []
            cur = v
            while cur not in nodes:
                if cur not in positions:
                    break
                chain.append(positions[cur])
                if cur not in forward:
                    break
                cur = forward[cur]
            if cur in nodes and u in positions and cur in positions:
                polylines[(u, cur)] = [positions[u], *chain, positions[cur]]
        return polylines

    @staticmethod
    def _clip_point_to_aabb(
        center: tuple[float, float],
        half: tuple[float, float],
        toward: tuple[float, float],
    ) -> tuple[float, float]:
        cx, cy = center
        hw, hh = half
        dx = toward[0] - cx
        dy = toward[1] - cy
        if dx == 0.0 and dy == 0.0:
            return center
        tx = hw / abs(dx) if dx != 0.0 else float("inf")
        ty = hh / abs(dy) if dy != 0.0 else float("inf")
        t = min(tx, ty, 1.0)
        return (cx + t * dx, cy + t * dy)

    @classmethod
    def _clip_edge_polyline(
        cls,
        points: list[tuple[float, float]],
        src_node: GraphNode,
        tgt_node: GraphNode,
    ) -> list[tuple[float, float]]:
        if len(points) < 2:
            return points
        clipped = list(points)
        # Edges exit the source from its bottom-midpoint and enter the
        # target at its top-midpoint. Predictable anchors keep parallel
        # edges docked together and avoid endpoint/box intersections.
        clipped[0] = (
            float(src_node.x),
            float(src_node.y) + float(src_node.height) / 2.0,
        )
        clipped[-1] = (
            float(tgt_node.x),
            float(tgt_node.y) - float(tgt_node.height) / 2.0,
        )
        return clipped

    @staticmethod
    def _segment_crosses_aabb(
        p1: tuple[float, float],
        p2: tuple[float, float],
        aabb_center: tuple[float, float],
        aabb_half: tuple[float, float],
    ) -> bool:
        # Liang-Barsky style parametric slab test in the AABB's local frame.
        cx, cy = aabb_center
        hx, hy = aabb_half
        x1, y1 = p1[0] - cx, p1[1] - cy
        x2, y2 = p2[0] - cx, p2[1] - cy
        dx, dy = x2 - x1, y2 - y1
        t_enter, t_exit = 0.0, 1.0
        for p, q in ((-dx, x1 + hx), (dx, hx - x1), (-dy, y1 + hy), (dy, hy - y1)):
            if p == 0.0:
                if q < 0.0:
                    return False
                continue
            t = q / p
            if p < 0.0:
                if t > t_exit:
                    return False
                if t > t_enter:
                    t_enter = t
            else:
                if t < t_enter:
                    return False
                if t < t_exit:
                    t_exit = t
        return t_enter < t_exit

    def _compute_layout(self, nodes: dict[str, GraphNode], edges: list[GraphEdge]) -> None:
        ext_label_lines_by_node: dict[str, list[str]] = {}
        for node_id in nodes:
            ext_lines: list[str] = []
            for ext in self.extensions:
                ext_lines.extend(self._ext_label_lines_for_layout(ext, node_id))
            ext_label_lines_by_node[node_id] = ext_lines

        self._compute_layout_with_ext_lines(
            nodes,
            edges,
            ext_label_lines_by_node=ext_label_lines_by_node,
            base_label_getter=self._safe_base_label,
        )

    @staticmethod
    def _coerce_str_lines(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [x for x in value if isinstance(x, str)]

    @classmethod
    def relayout_payload_base(
        cls,
        base_payload: Dict[str, Any],
        extensions_payload: Optional[Dict[str, Any]] = None,
        include_layers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Recompute base graph node/edge layout using extension label lines.

        This API operates on payload dictionaries and does not require
        a ``torch.fx.GraphModule`` instance.
        """
        if not isinstance(base_payload, dict):
            raise TypeError("base_payload must be a dict")

        relaid = copy.deepcopy(base_payload)

        raw_nodes = relaid.get("nodes")
        raw_edges = relaid.get("edges")
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise ValueError("base_payload must contain list fields: 'nodes' and 'edges'")

        nodes: dict[str, GraphNode] = {}
        for idx, node_data in enumerate(raw_nodes):
            if not isinstance(node_data, dict):
                continue
            node_id = str(node_data.get("id", "")).strip()
            if not node_id:
                continue

            info_value = node_data.get("info", {})
            tooltip_value = node_data.get("tooltip", [])
            node = GraphNode(
                id=node_id,
                label=str(node_data.get("label", "")),
                topo_index=int(node_data.get("topo_index", idx)),
                info=info_value if isinstance(info_value, dict) else {},
                tooltip=tooltip_value if isinstance(tooltip_value, list) else [],
                fill_color=node_data.get("fill_color"),
            )
            nodes[node_id] = node

        edges: list[GraphEdge] = []
        for edge_data in raw_edges:
            if not isinstance(edge_data, dict):
                continue
            v = str(edge_data.get("v", "")).strip()
            w = str(edge_data.get("w", "")).strip()
            if not v or not w or v not in nodes or w not in nodes:
                continue
            edges.append(GraphEdge(v=v, w=w, points=[]))

        ext_payloads = extensions_payload if isinstance(extensions_payload, dict) else {}
        if include_layers is None:
            active_layer_ids = list(ext_payloads.keys())
        else:
            active_layer_ids = [layer_id for layer_id in include_layers if layer_id in ext_payloads]

        ext_label_lines_by_node: dict[str, list[str]] = {node_id: [] for node_id in nodes}
        for layer_id in active_layer_ids:
            layer_payload = ext_payloads.get(layer_id)
            if not isinstance(layer_payload, dict):
                continue
            layer_nodes = layer_payload.get("nodes")
            if not isinstance(layer_nodes, dict):
                continue
            for node_id, node_payload in layer_nodes.items():
                if node_id not in ext_label_lines_by_node or not isinstance(node_payload, dict):
                    continue
                ext_label_lines_by_node[node_id].extend(
                    cls._coerce_str_lines(node_payload.get("label_append"))
                )

        cls._compute_layout_with_ext_lines(
            nodes,
            edges,
            ext_label_lines_by_node=ext_label_lines_by_node,
            base_label_getter=lambda node: str(node.label or ""),
        )

        node_by_id = {node.get("id"): node for node in raw_nodes if isinstance(node, dict)}
        for node_id, node in nodes.items():
            node_dict = node_by_id.get(node_id)
            if not isinstance(node_dict, dict):
                continue
            node_dict["x"] = node.x
            node_dict["y"] = node.y
            node_dict["width"] = node.width
            node_dict["height"] = node.height

        edge_points_by_key = {(edge.v, edge.w): edge.points for edge in edges}
        for edge_data in raw_edges:
            if not isinstance(edge_data, dict):
                continue
            key = (
                str(edge_data.get("v", "")).strip(),
                str(edge_data.get("w", "")).strip(),
            )
            edge_data["points"] = edge_points_by_key.get(key, [])

        return relaid

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
