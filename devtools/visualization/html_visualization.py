#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
"""Generate an interactive HTML visualization of an ExecuTorch graph.

Supports pre-serialization (.pt2), post-serialization (.pte), ETRecord
(.etrecord), and multi-pass trace (.json) files. Produces a self-contained
HTML file using Cytoscape.js with dagre layout.

Usage:
    python3 -m executorch.devtools.visualization.html_visualization model.pt2
    python3 -m executorch.devtools.visualization.html_visualization model.pte -o graph.html
    python3 -m executorch.devtools.visualization.html_visualization trace.json -o passes.html

Authored with Claude.
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional


CATEGORY_COLORS = {
    "backend": "#4caf50",
    "aten_compute": "#2196f3",
    "quantize": "#ff9800",
    "memory": "#9e9e9e",
    "placeholder": "#03a9f4",
    "param": "#78909c",
    "delegate": "#ab47bc",
}

# Backend custom-op prefixes. Ops containing these (case-insensitive) are
# categorized as "backend" rather than generic compute. Extend this tuple
# when new backends register custom op namespaces.
_BACKEND_OP_PREFIXES = (
    "cortex_m",
    "cadence",
    "qaisw",
)


def categorize_node(op_name: str) -> str:
    name = op_name.lower()
    if any(prefix in name for prefix in _BACKEND_OP_PREFIXES):
        return "backend"
    if any(
        k in name
        for k in (
            "quantize_per_tensor",
            "dequantize_per_",
            "quantize_per_channel",
            "dequantize_per_channel",
        )
    ):
        return "quantize"
    if any(
        k in name
        for k in (
            "view",
            "clone",
            "permute",
            "slice",
            "copy",
            "expand",
            "reshape",
            "t_copy",
            "unsqueeze",
            "squeeze",
        )
    ):
        return "memory"
    if any(k in name for k in ("placeholder", "output", "getitem", "get_attr")):
        return "placeholder"
    if "delegate" in name:
        return "delegate"
    return "aten_compute"


def _make_label(op_name: str) -> str:
    name = op_name.split("::")[-1] if "::" in op_name else op_name
    if "." in name:
        name = name.rsplit(".", 1)[0]
    if len(name) > 30:
        name = name[:27] + "..."
    return name


def extract_from_exported_program(ep, model_name: str) -> dict:
    """Walk an in-memory ExportedProgram's graph and extract visualization data."""
    graph = ep.graph

    nodes = []
    edges = []
    node_map = {}

    for node in graph.nodes:
        node_id = node.name
        op_name = node.op
        if node.op == "call_function":
            op_name = str(node.target)
        elif node.op == "call_method":
            op_name = node.target

        details = {"op": node.op, "target": str(getattr(node, "target", ""))}
        meta_val = node.meta.get("val")
        if meta_val is not None:
            if hasattr(meta_val, "shape"):
                details["shape"] = str(list(meta_val.shape))
                details["dtype"] = str(meta_val.dtype)
            elif isinstance(meta_val, (list, tuple)):
                shapes = []
                for v in meta_val:
                    if hasattr(v, "shape"):
                        shapes.append(f"{list(v.shape)} {v.dtype}")
                if shapes:
                    details["shapes"] = shapes

        category = categorize_node(op_name)
        label = _make_label(op_name)

        if node.op == "placeholder":
            target = str(getattr(node, "target", ""))
            if target.startswith("p_") and target.endswith("_weight"):
                label = "weight"
                category = "param"
            elif target.startswith("p_") and target.endswith("_bias"):
                label = "bias"
                category = "param"
            elif target.startswith("b_") and "running_mean" in target:
                label = "bn_mean"
                category = "param"
            elif target.startswith("b_") and "running_var" in target:
                label = "bn_var"
                category = "param"
            elif target == "x" or not target.startswith(("p_", "b_")):
                label = "input"
        elif node.op == "output":
            label = "output"

        node_map[node_id] = len(nodes)
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "w": max(len(label) * 8 + 16, 60),
                "category": category,
                "op_name": op_name,
                "details": details,
            }
        )

        for arg in node.args:
            if hasattr(arg, "name") and arg.name in node_map:
                edges.append({"source": arg.name, "target": node_id})
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    if hasattr(a, "name") and a.name in node_map:
                        edges.append({"source": a.name, "target": node_id})

    category_counts = {}
    for n in nodes:
        cat = n["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "metadata": {
            "model_name": model_name,
            "source_type": "pt2",
            "total_nodes": len(nodes),
            "category_counts": category_counts,
        },
        "nodes": nodes,
        "edges": edges,
    }


def extract_from_pt2(path: str) -> dict:
    import torch

    ep = torch.export.load(path)
    return extract_from_exported_program(ep, os.path.basename(path))


def _extract_delegate_blob_info(
    delegate_data: bytes,
) -> Optional[Dict[str, Any]]:
    """Extract metadata from a delegate blob. Backend-agnostic."""
    if len(delegate_data) < 8:
        return None
    info: Dict[str, Any] = {"blob_size_bytes": len(delegate_data)}
    op_patterns = re.findall(rb"(?:tosa|aten|xnnpack|qnn)\.\w+", delegate_data)
    if op_patterns:
        info["detected_ops"] = sorted(set(op.decode() for op in op_patterns[:20]))
    return info


def extract_from_pte(path: str) -> dict:
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import DelegateCall, KernelCall, Tensor

    with open(path, "rb") as f:
        data = f.read()

    pte_file = deserialize_pte_binary(data)
    plan = pte_file.program.execution_plan[0]

    nodes: List[dict] = []
    edges: List[dict] = []
    value_producers: Dict[int, str] = {}

    # Extract delegate blobs for analysis
    delegate_info_map: Dict[int, dict] = {}
    for idx, delegate in enumerate(plan.delegates):
        if hasattr(delegate, "processed") and delegate.processed:
            blob = getattr(delegate.processed, "data", None)
            if blob:
                info = _extract_delegate_blob_info(bytes(blob))
                if info:
                    delegate_info_map[idx] = info

    for chain_idx, chain in enumerate(plan.chains):
        for instr_idx, instr in enumerate(chain.instructions):
            args = instr.instr_args

            if isinstance(args, KernelCall):
                op = plan.operators[args.op_index]
                op_name = f"{op.name}.{op.overload}" if op.overload else op.name
                node_id = f"k_{chain_idx}_{instr_idx}"

                details: Dict[str, Any] = {"op_name": op_name}
                input_tensors = []
                output_tensors = []
                for val_idx in args.args:
                    if val_idx < len(plan.values):
                        val = plan.values[val_idx].val
                        if isinstance(val, Tensor):
                            shape_str = (
                                f"[{','.join(str(s) for s in val.sizes)}]"
                            )
                            dtype_str = (
                                val.scalar_type.name
                                if hasattr(val.scalar_type, "name")
                                else str(val.scalar_type)
                            )
                            info_str = f"{shape_str} {dtype_str}"
                            if val_idx in value_producers:
                                input_tensors.append(info_str)
                            else:
                                output_tensors.append(info_str)

                if input_tensors:
                    details["inputs"] = input_tensors
                if output_tensors:
                    details["outputs"] = output_tensors

                category = categorize_node(op_name)
                label = _make_label(op_name)

                nodes.append(
                    {
                        "id": node_id,
                        "label": label,
                        "w": max(len(label) * 8 + 16, 60),
                        "category": category,
                        "op_name": op_name,
                        "details": details,
                    }
                )

                for val_idx in args.args:
                    if val_idx in value_producers:
                        edges.append(
                            {"source": value_producers[val_idx], "target": node_id}
                        )

                for val_idx in args.args:
                    value_producers[val_idx] = node_id

            elif isinstance(args, DelegateCall):
                node_id = f"d_{chain_idx}_{instr_idx}"
                delegate = plan.delegates[args.delegate_index]

                details = {
                    "delegate_id": delegate.id,
                    "delegate_index": args.delegate_index,
                }
                if args.delegate_index in delegate_info_map:
                    details.update(delegate_info_map[args.delegate_index])

                label = delegate.id
                if len(label) > 25:
                    label = label[:22] + "..."

                nodes.append(
                    {
                        "id": node_id,
                        "label": label,
                        "w": max(len(label) * 8 + 20, 100),
                        "category": "delegate",
                        "op_name": f"delegate:{delegate.id}",
                        "details": details,
                    }
                )

                for val_idx in args.args:
                    if val_idx in value_producers:
                        edges.append(
                            {"source": value_producers[val_idx], "target": node_id}
                        )

                for val_idx in args.args:
                    value_producers[val_idx] = node_id

    for i, idx in enumerate(plan.inputs):
        node_id = f"input_{i}"
        val = plan.values[idx].val
        details = {"value_index": idx}
        if isinstance(val, Tensor):
            details["shape"] = list(val.sizes)
            details["dtype"] = (
                val.scalar_type.name
                if hasattr(val.scalar_type, "name")
                else str(val.scalar_type)
            )

        nodes.insert(
            0,
            {
                "id": node_id,
                "label": f"input_{i}",
                "w": 70,
                "category": "placeholder",
                "op_name": "input",
                "details": details,
            },
        )
        value_producers[idx] = node_id

    for i, idx in enumerate(plan.outputs):
        node_id = f"output_{i}"
        val = plan.values[idx].val
        details = {"value_index": idx}
        if isinstance(val, Tensor):
            details["shape"] = list(val.sizes)
            details["dtype"] = (
                val.scalar_type.name
                if hasattr(val.scalar_type, "name")
                else str(val.scalar_type)
            )
        nodes.append(
            {
                "id": node_id,
                "label": f"output_{i}",
                "w": 80,
                "category": "placeholder",
                "op_name": "output",
                "details": details,
            }
        )
        if idx in value_producers:
            edges.append({"source": value_producers[idx], "target": node_id})

    # Filter edges to only reference existing nodes
    node_ids = {n["id"] for n in nodes}
    edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]

    category_counts = {}
    for n in nodes:
        cat = n["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "metadata": {
            "model_name": os.path.basename(path),
            "source_type": "pte",
            "total_nodes": len(nodes),
            "category_counts": category_counts,
        },
        "nodes": nodes,
        "edges": edges,
    }


def extract_from_trace_json(path: str) -> dict:
    """Load a multi-pass trace JSON."""
    with open(path) as f:
        data = json.load(f)
    if "passes" not in data:
        raise ValueError(f"{path} does not contain a 'passes' key")
    return data


def extract_from_etrecord(path: str) -> dict:
    """Extract visualization data from an ETRecord file."""
    from executorch.devtools.etrecord import parse_etrecord

    etrecord = parse_etrecord(path)
    passes = []

    # edge_dialect_program can be a single ExportedProgram or a dict of them
    edp = etrecord.edge_dialect_program
    if edp is not None:
        if isinstance(edp, dict):
            for method_name, ep in edp.items():
                passes.append(
                    extract_from_exported_program(ep, f"Edge Dialect: {method_name}")
                )
        else:
            passes.append(
                extract_from_exported_program(edp, "Edge Dialect (pre-delegation)")
            )

    # graph_map contains additional stages
    if etrecord.graph_map:
        for name, ep in etrecord.graph_map.items():
            passes.append(extract_from_exported_program(ep, name))

    # Fallback to exported_program if nothing else is available
    if etrecord.exported_program is not None and not passes:
        passes.append(
            extract_from_exported_program(etrecord.exported_program, "Exported Program")
        )

    if len(passes) == 0:
        raise ValueError(f"No graph data found in {path}")

    if len(passes) == 1:
        return passes[0]

    return {
        "model_name": os.path.basename(path),
        "passes": passes,
    }


# ---------------------------------------------------------------------------
# Single-pass HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ExecuTorch Graph: $$MODEL_NAME$$</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; }
#header { padding: 10px 16px; background: #16213e; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; border-bottom: 1px solid #0f3460; }
#header h1 { font-size: 16px; font-weight: 600; white-space: nowrap; }
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; }
.badge .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
#container { display: flex; height: calc(100vh - 45px); position: relative; }
#cy { flex: 1; min-height: 400px; }
#controls { position: absolute; top: 10px; left: 10px; display: flex; flex-direction: column; gap: 4px; z-index: 10; }
#controls button { width: 36px; height: 36px; border: 1px solid #0f3460; border-radius: 6px; background: #16213e; color: #eee; font-size: 18px; cursor: pointer; }
#controls button:hover { background: #0f3460; }
#panel { width: 340px; background: #16213e; border-left: 1px solid #0f3460; padding: 16px; overflow-y: auto; display: none; }
#panel h2 { font-size: 14px; margin-bottom: 12px; color: #e94560; }
#panel .field { margin-bottom: 8px; }
#panel .field-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
#panel .field-value { font-size: 13px; word-break: break-all; margin-top: 2px; }
#panel .close-btn { float: right; cursor: pointer; color: #888; font-size: 18px; }
#panel .close-btn:hover { color: #fff; }
#error-msg { padding: 20px; color: #e94560; display: none; }
</style>
</head>
<body>
<div id="header">
  <h1>ExecuTorch Graph: <span id="model-name"></span></h1>
  <span class="badge" style="background:#333">Source: <span id="source-type"></span></span>
  <span class="badge" style="background:#333">Nodes: <span id="total-nodes"></span></span>
  <span id="category-badges"></span>
</div>
<div id="error-msg"></div>
<div id="container">
  <div id="controls">
    <button onclick="graphFit()" title="Fit all">F</button>
    <button onclick="graphZoom(1.5)" title="Zoom in">+</button>
    <button onclick="graphZoom(0.67)" title="Zoom out">&minus;</button>
  </div>
  <div id="cy"></div>
  <div id="panel">
    <span class="close-btn" onclick="document.getElementById('panel').style.display='none'">&times;</span>
    <h2 id="panel-title"></h2>
    <div id="panel-content"></div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.3.2/cytoscape-dagre.min.js"></script>
<script>
var GRAPH_DATA = $$GRAPH_JSON$$;
var COLORS = $$COLORS_JSON$$;
var CATEGORY_LABELS = {
  backend: "Backend", aten_compute: "Compute", quantize: "Quant",
  memory: "Memory", placeholder: "I/O", param: "Params", delegate: "Delegate"
};
var _cy = null;

function escapeHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
function graphFit() { if (_cy) _cy.fit(null, 40); }
function graphZoom(factor) {
  if (!_cy) return;
  _cy.zoom({ level: _cy.zoom() * factor, renderedPosition: { x: _cy.width()/2, y: _cy.height()/2 } });
}

function initGraph() {
  try {
    if (typeof cytoscape === "undefined") {
      document.getElementById("error-msg").textContent = "Failed to load Cytoscape.js";
      document.getElementById("error-msg").style.display = "block";
      return;
    }

    document.getElementById("model-name").textContent = GRAPH_DATA.metadata.model_name;
    document.getElementById("source-type").textContent = GRAPH_DATA.metadata.source_type;
    document.getElementById("total-nodes").textContent = GRAPH_DATA.metadata.total_nodes;

    var badgesHtml = "";
    var cats = GRAPH_DATA.metadata.category_counts;
    for (var cat in cats) {
      var color = COLORS[cat] || "#666";
      var label = CATEGORY_LABELS[cat] || cat;
      badgesHtml += '<span class="badge" style="background:#333"><span class="dot" style="background:' + color + '"></span>' + escapeHtml(label) + ': ' + cats[cat] + '</span> ';
    }
    document.getElementById("category-badges").innerHTML = badgesHtml;

    var elements = [];
    for (var i = 0; i < GRAPH_DATA.nodes.length; i++) {
      var n = GRAPH_DATA.nodes[i];
      elements.push({ data: { id: n.id, label: n.label, w: n.w, category: n.category, color: COLORS[n.category] || "#666", op_name: n.op_name, details: n.details } });
    }
    for (var j = 0; j < GRAPH_DATA.edges.length; j++) {
      var e = GRAPH_DATA.edges[j];
      elements.push({ data: { source: e.source, target: e.target } });
    }

    var layoutName = (typeof dagre !== "undefined") ? "dagre" : "breadthfirst";

    _cy = cytoscape({
      container: document.getElementById("cy"),
      elements: elements,
      style: [
        { selector: "node", style: {
          "label": "data(label)",
          "background-color": "data(color)",
          "color": "#fff",
          "font-size": 12,
          "font-weight": "bold",
          "text-valign": "center",
          "text-halign": "center",
          "shape": "roundrectangle",
          "width": "data(w)",
          "height": 32,
          "border-width": 0
        }},
        { selector: "node:selected", style: { "border-width": 3, "border-color": "#e94560" }},
        { selector: "edge", style: {
          "width": 2,
          "line-color": "#555",
          "target-arrow-color": "#555",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          "arrow-scale": 0.8
        }},
        { selector: "edge:selected", style: { "line-color": "#e94560", "target-arrow-color": "#e94560" }}
      ],
      layout: { name: layoutName, rankDir: "TB", nodeSep: 50, rankSep: 60, edgeSep: 20, directed: true, spacingFactor: 1.5 },
      minZoom: 0.02,
      maxZoom: 4
    });

    _cy.on("tap", "node", function(evt) {
      var d = evt.target.data();
      var panel = document.getElementById("panel");
      document.getElementById("panel-title").textContent = d.op_name;
      var html = '<div class="field"><span class="field-label">Category</span><div class="field-value"><span class="dot" style="background:' + (d.color||"#666") + ';display:inline-block;width:10px;height:10px;border-radius:50%"></span> ' + escapeHtml(CATEGORY_LABELS[d.category] || d.category) + '</div></div>';
      html += '<div class="field"><span class="field-label">Node ID</span><div class="field-value">' + escapeHtml(d.id) + '</div></div>';
      if (d.details) {
        for (var k in d.details) {
          var val = d.details[k];
          if (typeof val === "object" && val !== null && !Array.isArray(val)) {
            val = JSON.stringify(val, null, 2);
            html += '<div class="field"><span class="field-label">' + escapeHtml(k) + '</span><div class="field-value"><pre style="white-space:pre-wrap;font-size:11px;color:#ccc">' + escapeHtml(val) + '</pre></div></div>';
          } else {
            val = Array.isArray(val) ? val.map(escapeHtml).join("<br>") : escapeHtml(String(val));
            html += '<div class="field"><span class="field-label">' + escapeHtml(k) + '</span><div class="field-value">' + val + '</div></div>';
          }
        }
      }
      document.getElementById("panel-content").innerHTML = html;
      panel.style.display = "block";
    });

    _cy.on("tap", function(evt) {
      if (evt.target === _cy) document.getElementById("panel").style.display = "none";
    });

  } catch(err) {
    document.getElementById("error-msg").textContent = "Error: " + err.message;
    document.getElementById("error-msg").style.display = "block";
    console.error(err);
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initGraph);
} else {
  initGraph();
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Multi-pass HTML template
# ---------------------------------------------------------------------------

MULTI_PASS_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ExecuTorch Pass Trace: $$MODEL_NAME$$</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; }
#header { padding: 10px 16px; background: #16213e; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; border-bottom: 1px solid #0f3460; }
#header h1 { font-size: 16px; font-weight: 600; white-space: nowrap; }
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; }
.badge .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
#pass-bar { padding: 8px 16px; background: #1a1a3e; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; border-bottom: 1px solid #0f3460; }
#pass-bar select { background: #16213e; color: #eee; border: 1px solid #0f3460; border-radius: 4px; padding: 4px 8px; font-size: 13px; min-width: 240px; }
#pass-bar button { width: 32px; height: 32px; border: 1px solid #0f3460; border-radius: 6px; background: #16213e; color: #eee; font-size: 16px; cursor: pointer; }
#pass-bar button:hover { background: #0f3460; }
#pass-bar button:disabled { opacity: 0.3; cursor: default; }
#node-count-badge { font-size: 12px; color: #aaa; }
#diff-badge { font-size: 12px; font-weight: 600; }
#error-banner { padding: 10px 16px; background: #5c1010; color: #ff6b6b; display: none; border-bottom: 1px solid #8b0000; }
#error-banner .error-title { font-weight: 600; margin-bottom: 4px; }
#error-banner .error-msg { font-size: 13px; }
#error-banner details { margin-top: 6px; font-size: 12px; }
#error-banner summary { cursor: pointer; color: #ff9999; }
#error-banner pre { margin-top: 4px; white-space: pre-wrap; font-size: 11px; color: #ffaaaa; max-height: 300px; overflow-y: auto; }
#container { display: flex; height: calc(100vh - 95px); position: relative; }
#cy { flex: 1; min-height: 400px; }
#controls { position: absolute; top: 10px; left: 10px; display: flex; flex-direction: column; gap: 4px; z-index: 10; }
#controls button { width: 36px; height: 36px; border: 1px solid #0f3460; border-radius: 6px; background: #16213e; color: #eee; font-size: 18px; cursor: pointer; }
#controls button:hover { background: #0f3460; }
#panel { width: 380px; background: #16213e; border-left: 1px solid #0f3460; padding: 16px; overflow-y: auto; display: none; }
#panel h2 { font-size: 14px; margin-bottom: 12px; color: #e94560; }
#panel .field { margin-bottom: 8px; }
#panel .field-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
#panel .field-value { font-size: 13px; word-break: break-all; margin-top: 2px; }
#panel pre { white-space: pre-wrap; font-size: 11px; color: #ccc; }
#panel .close-btn { float: right; cursor: pointer; color: #888; font-size: 18px; }
#panel .close-btn:hover { color: #fff; }
#category-badges { display: inline-flex; gap: 4px; flex-wrap: wrap; }
</style>
</head>
<body>
<div id="header">
  <h1>ExecuTorch Pass Trace: <span id="model-name">$$MODEL_NAME$$</span></h1>
  <span class="badge" style="background:#333">Passes: <span id="pass-count"></span></span>
  <span id="category-badges"></span>
</div>
<div id="pass-bar">
  <button id="btn-prev" onclick="prevPass()" title="Previous pass">&larr;</button>
  <select id="pass-select" onchange="switchToPass(parseInt(this.value))"></select>
  <button id="btn-next" onclick="nextPass()" title="Next pass">&rarr;</button>
  <span class="badge" style="background:#333">Nodes: <span id="node-count-badge">0</span></span>
  <span id="diff-badge"></span>
</div>
<div id="error-banner">
  <div class="error-title" id="error-title"></div>
  <div class="error-msg" id="error-msg-text"></div>
  <details><summary>Traceback</summary><pre id="error-traceback"></pre></details>
</div>
<div id="container">
  <div id="controls">
    <button onclick="graphFit()" title="Fit all">F</button>
    <button onclick="graphZoom(1.5)" title="Zoom in">+</button>
    <button onclick="graphZoom(0.67)" title="Zoom out">&minus;</button>
  </div>
  <div id="cy"></div>
  <div id="panel">
    <span class="close-btn" onclick="document.getElementById('panel').style.display='none'">&times;</span>
    <h2 id="panel-title"></h2>
    <div id="panel-content"></div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.3.2/cytoscape-dagre.min.js"></script>
<script>
var ALL_PASSES = $$PASSES_JSON$$;
var COLORS = $$COLORS_JSON$$;
var CATEGORY_LABELS = {
  backend: "Backend", aten_compute: "Compute", quantize: "Quant",
  memory: "Memory", placeholder: "I/O", param: "Params", delegate: "Delegate"
};
var QDQ_GROUP_COLORS = ["#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#fabebe","#469990","#dcbeff","#9a6324","#800000","#aaffc3","#808000","#ffd8b1","#000075"];
var currentPassIndex = 0;
var _cy = null;

function graphFit() { if (_cy) _cy.fit(null, 40); }
function graphZoom(factor) {
  if (!_cy) return;
  _cy.zoom({ level: _cy.zoom() * factor, renderedPosition: { x: _cy.width()/2, y: _cy.height()/2 } });
}
function prevPass() { if (currentPassIndex > 0) switchToPass(currentPassIndex - 1); }
function nextPass() { if (currentPassIndex < ALL_PASSES.length - 1) switchToPass(currentPassIndex + 1); }

function escapeHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function renderDetailValue(key, val) {
  if (typeof val === "object" && val !== null && !Array.isArray(val)) {
    return '<div class="field"><span class="field-label">' + escapeHtml(key) + '</span><div class="field-value"><pre>' + escapeHtml(JSON.stringify(val, null, 2)) + '</pre></div></div>';
  }
  var display = Array.isArray(val) ? val.map(escapeHtml).join("<br>") : escapeHtml(String(val));
  return '<div class="field"><span class="field-label">' + escapeHtml(key) + '</span><div class="field-value">' + display + '</div></div>';
}

function switchToPass(index) {
  currentPassIndex = index;
  var passData = ALL_PASSES[index];
  document.getElementById("pass-select").value = index;
  document.getElementById("btn-prev").disabled = (index === 0);
  document.getElementById("btn-next").disabled = (index === ALL_PASSES.length - 1);

  var nodeCount = passData.metadata.total_nodes;
  document.getElementById("node-count-badge").textContent = nodeCount;
  var diffBadge = document.getElementById("diff-badge");
  if (index > 0) {
    var prevCount = ALL_PASSES[index - 1].metadata.total_nodes;
    var delta = nodeCount - prevCount;
    if (delta > 0) {
      diffBadge.textContent = "+" + delta + " nodes";
      diffBadge.style.color = "#4caf50";
    } else if (delta < 0) {
      diffBadge.textContent = delta + " nodes";
      diffBadge.style.color = "#e94560";
    } else {
      diffBadge.textContent = "no change";
      diffBadge.style.color = "#888";
    }
  } else {
    diffBadge.textContent = "";
  }

  var badgesHtml = "";
  var cats = passData.metadata.category_counts;
  for (var cat in cats) {
    var color = COLORS[cat] || "#666";
    var label = CATEGORY_LABELS[cat] || cat;
    badgesHtml += '<span class="badge" style="background:#333"><span class="dot" style="background:' + color + '"></span>' + escapeHtml(label) + ': ' + cats[cat] + '</span> ';
  }
  document.getElementById("category-badges").innerHTML = badgesHtml;

  var errorBanner = document.getElementById("error-banner");
  var err = passData.metadata.error;
  if (err) {
    document.getElementById("error-title").textContent = "Error in " + err.pass_name;
    document.getElementById("error-msg-text").textContent = err.message;
    document.getElementById("error-traceback").textContent = err.traceback || "";
    errorBanner.style.display = "block";
  } else {
    errorBanner.style.display = "none";
  }

  document.getElementById("panel").style.display = "none";

  var hasQdqGroups = false;
  var qdqGroupSet = {};
  for (var i = 0; i < passData.nodes.length; i++) {
    if (passData.nodes[i].qdq_group_id !== undefined && passData.nodes[i].qdq_group_id !== null) {
      hasQdqGroups = true;
      qdqGroupSet[passData.nodes[i].qdq_group_id] = true;
    }
  }

  var elements = [];

  if (hasQdqGroups) {
    for (var gid in qdqGroupSet) {
      elements.push({
        data: { id: "qdq_group_" + gid, label: "QDQ Group " + gid },
        classes: "qdq-parent"
      });
    }
  }

  for (var i = 0; i < passData.nodes.length; i++) {
    var n = passData.nodes[i];
    var nodeData = {
      id: n.id, label: n.label, w: n.w,
      category: n.category,
      color: COLORS[n.category] || "#666",
      op_name: n.op_name, details: n.details
    };
    if (hasQdqGroups && n.qdq_group_id !== undefined && n.qdq_group_id !== null) {
      nodeData.parent = "qdq_group_" + n.qdq_group_id;
    }
    elements.push({ data: nodeData });
  }
  for (var j = 0; j < passData.edges.length; j++) {
    var e = passData.edges[j];
    elements.push({ data: { source: e.source, target: e.target } });
  }

  if (_cy) _cy.destroy();
  var layoutName = (typeof dagre !== "undefined") ? "dagre" : "breadthfirst";

  var styles = [
    { selector: "node", style: {
      "label": "data(label)",
      "background-color": "data(color)",
      "color": "#fff",
      "font-size": 12,
      "font-weight": "bold",
      "text-valign": "center",
      "text-halign": "center",
      "shape": "roundrectangle",
      "width": "data(w)",
      "height": 32,
      "border-width": 0
    }},
    { selector: "node:selected", style: { "border-width": 3, "border-color": "#e94560" }},
    { selector: "edge", style: {
      "width": 2,
      "line-color": "#555",
      "target-arrow-color": "#555",
      "target-arrow-shape": "triangle",
      "curve-style": "bezier",
      "arrow-scale": 0.8
    }},
    { selector: "edge:selected", style: { "line-color": "#e94560", "target-arrow-color": "#e94560" }}
  ];

  if (hasQdqGroups) {
    styles.push({
      selector: ".qdq-parent", style: {
        "background-opacity": 0.08,
        "border-width": 2,
        "border-style": "dashed",
        "border-color": "#888",
        "label": "data(label)",
        "color": "#aaa",
        "font-size": 10,
        "text-valign": "top",
        "text-halign": "center",
        "shape": "roundrectangle",
        "padding": "12px"
      }
    });
    for (var gid in qdqGroupSet) {
      var groupColor = QDQ_GROUP_COLORS[parseInt(gid) % QDQ_GROUP_COLORS.length];
      styles.push({
        selector: '#qdq_group_' + gid, style: { "border-color": groupColor }
      });
    }
  }

  _cy = cytoscape({
    container: document.getElementById("cy"),
    elements: elements,
    style: styles,
    layout: { name: layoutName, rankDir: "TB", nodeSep: 50, rankSep: 60, edgeSep: 20, directed: true, spacingFactor: 1.5 },
    minZoom: 0.02,
    maxZoom: 4
  });

  _cy.on("tap", "node", function(evt) {
    if (evt.target.hasClass && evt.target.hasClass("qdq-parent")) return;
    var d = evt.target.data();
    var panel = document.getElementById("panel");
    document.getElementById("panel-title").textContent = d.op_name || d.label;
    var html = '';
    if (d.category) {
      html += '<div class="field"><span class="field-label">Category</span><div class="field-value"><span class="dot" style="background:' + (d.color||"#666") + ';display:inline-block;width:10px;height:10px;border-radius:50%"></span> ' + escapeHtml(CATEGORY_LABELS[d.category] || d.category) + '</div></div>';
    }
    html += '<div class="field"><span class="field-label">Node ID</span><div class="field-value">' + escapeHtml(d.id) + '</div></div>';
    if (d.details) {
      for (var k in d.details) {
        html += renderDetailValue(k, d.details[k]);
      }
    }
    document.getElementById("panel-content").innerHTML = html;
    panel.style.display = "block";
  });

  _cy.on("tap", function(evt) {
    if (evt.target === _cy) document.getElementById("panel").style.display = "none";
  });
}

function initMultiPass() {
  if (typeof cytoscape === "undefined") {
    document.getElementById("error-banner").style.display = "block";
    document.getElementById("error-title").textContent = "Failed to load Cytoscape.js";
    return;
  }

  document.getElementById("pass-count").textContent = ALL_PASSES.length;

  var select = document.getElementById("pass-select");
  for (var i = 0; i < ALL_PASSES.length; i++) {
    var opt = document.createElement("option");
    opt.value = i;
    var name = ALL_PASSES[i].metadata.model_name;
    var hasError = ALL_PASSES[i].metadata.error;
    opt.textContent = (hasError ? "[ERR] " : "") + name + " (" + ALL_PASSES[i].metadata.total_nodes + " nodes)";
    if (hasError) opt.style.color = "#ff6b6b";
    select.appendChild(opt);
  }

  document.addEventListener("keydown", function(e) {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.key === "ArrowLeft") prevPass();
    else if (e.key === "ArrowRight") nextPass();
  });

  switchToPass(0);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initMultiPass);
} else {
  initMultiPass();
}
</script>
</body>
</html>"""


def generate_html(graph_data: dict, output_path: str) -> None:
    html = HTML_TEMPLATE
    html = html.replace("$$MODEL_NAME$$", graph_data["metadata"]["model_name"])
    html = html.replace("$$GRAPH_JSON$$", json.dumps(graph_data))
    html = html.replace("$$COLORS_JSON$$", json.dumps(CATEGORY_COLORS))
    with open(output_path, "w") as f:
        f.write(html)
    print(
        f"Wrote {output_path} "
        f"({graph_data['metadata']['total_nodes']} nodes, "
        f"{len(graph_data['edges'])} edges)"
    )


def generate_multi_pass_html(trace_data: dict, output_path: str) -> None:
    model_name = trace_data.get("model_name", "unknown")
    passes = trace_data["passes"]

    html = MULTI_PASS_HTML_TEMPLATE
    html = html.replace("$$MODEL_NAME$$", model_name)
    html = html.replace("$$PASSES_JSON$$", json.dumps(passes))
    html = html.replace("$$COLORS_JSON$$", json.dumps(CATEGORY_COLORS))
    with open(output_path, "w") as f:
        f.write(html)

    total_nodes = sum(p["metadata"]["total_nodes"] for p in passes)
    print(
        f"Wrote {output_path} ({len(passes)} passes, "
        f"{total_nodes} total nodes across all snapshots)"
    )


def visualize_edge_manager(edge_manager, output_path: str = "graph.html") -> str:
    """Visualize an EdgeProgramManager as HTML before to_executorch().

    Usage in your export script:
        from executorch.devtools.visualization.html_visualization import (
            visualize_edge_manager,
        )

        edge_manager = to_edge_transform_and_lower(...)
        visualize_edge_manager(edge_manager, "my_model_graph.html")
        et_program = edge_manager.to_executorch()
    """
    ep = edge_manager.exported_program()
    graph_data = extract_from_exported_program(ep, "Edge Manager Graph")
    generate_html(graph_data, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ExecuTorch graph as interactive HTML"
    )
    parser.add_argument("input", help="Path to .pt2, .pte, .etrecord, or .json file")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    output = args.output or os.path.splitext(args.input)[0] + ".html"
    ext = os.path.splitext(args.input)[1].lower()

    if ext == ".json":
        trace_data = extract_from_trace_json(args.input)
        generate_multi_pass_html(trace_data, output)
    elif ext == ".pt2":
        graph_data = extract_from_pt2(args.input)
        generate_html(graph_data, output)
    elif ext == ".pte":
        graph_data = extract_from_pte(args.input)
        generate_html(graph_data, output)
    elif ext in (".etrecord", ".bin"):
        data = extract_from_etrecord(args.input)
        if "passes" in data:
            generate_multi_pass_html(data, output)
        else:
            generate_html(data, output)
    else:
        print(f"Error: unsupported '{ext}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
