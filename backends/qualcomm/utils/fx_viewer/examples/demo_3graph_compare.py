#!/usr/bin/env python3
"""Standalone 3-graph compare demo: Reference → Decomposed (1-to-many) → Fused (many-to-1).

Demonstrates from_node_root as the primary sync mode:
- Graph 1 (Reference):   torch.export float model; run_decompositions populates from_node.
- Graph 2 (Decomposed):  linear → t + mm + add (3 nodes share from_node_root="linear").
- Graph 3 (Fused):       deepcopy of ref_gm; relu nodes get union debug_handle for many-to-1.

Sync mode 'auto' (from_node_root → debug_handle → id) connects all three graphs:
- Click linear in Graph 1 → Graph 2: add_tensor (last of {t,mm,add} with from_node_root=linear).
- Click t_default/mm_default/add_tensor in Graph 2 → Graph 1: linear (from_node_root match).
- Click relu in Graph 1 → Graph 2: relu_default (from_node_root=relu).
- Click relu in Graph 3 (union handle) → Graph 1: relu. Graph 2: relu_default.

Run from repo root:
  python backends/qualcomm/utils/fx_viewer/examples/demo_3graph_compare.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.fx

from executorch.backends.qualcomm.utils.fx_viewer import (
    FXGraphExporter,
    GraphExtension,
    NumericColorRule,
)

THIS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _ToyModel(torch.nn.Module):
    """Small MLP — Linear+ReLU blocks are decomposed/fused to demo handle mapping."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# 1-to-many: DecomposeLinear transformer
# ---------------------------------------------------------------------------

class _DecomposeLinearPass(torch.fx.Transformer):
    """1-to-many: aten.linear → aten.t + aten.mm + aten.add (3 nodes, same handle)."""

    def call_function(self, target, args, kwargs):
        if target is torch.ops.aten.linear.default:
            inp, weight = args[0], args[1]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            t = super().call_function(torch.ops.aten.t.default, (weight,), {})
            mm = super().call_function(torch.ops.aten.mm.default, (inp, t), {})
            if bias is not None:
                return super().call_function(torch.ops.aten.add.Tensor, (mm, bias), {})
            return mm
        return super().call_function(target, args, kwargs)




# ---------------------------------------------------------------------------
# many-to-1: fused graph (no Transformer needed)
# ---------------------------------------------------------------------------

def _build_fused_graph(ref_gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Simulate fusion: relu nodes that follow a linear get a union handle (linear_h, relu_h).

    from_node is preserved from ref_gm via deepcopy, so from_node_root sync still works.
    The union debug_handle demonstrates many-to-1 matching as a secondary sync mechanism.
    """
    fused_gm = copy.deepcopy(ref_gm)
    nodes = list(fused_gm.graph.nodes)
    for i, node in enumerate(nodes):
        if node.op != "call_function" or "relu" not in node.name:
            continue
        prev = nodes[i - 1] if i > 0 else None
        if prev and prev.op == "call_function" and "linear" in prev.name:
            lh = prev.meta.get("debug_handle")
            rh = node.meta.get("debug_handle")
            if lh and rh:
                node.meta["debug_handle"] = (lh, rh)  # union tuple → many-to-1
    return fused_gm


# ---------------------------------------------------------------------------
# Extension builder
# ---------------------------------------------------------------------------

def _build_debug_handle_extension(graph_module: torch.fx.GraphModule) -> GraphExtension:
    ext = GraphExtension(id="debug_handle_sync", name="Debug Handle")
    for node in graph_module.graph.nodes:
        raw = node.meta.get("debug_handle")
        if not raw or raw == 0 or raw == () or raw == []:
            continue
        if isinstance(raw, int):
            dh_val: Any = raw
        elif isinstance(raw, (tuple, list)):
            ints = [int(x) for x in raw if isinstance(x, int) and x != 0]
            if not ints:
                continue
            dh_val = ints[0] if len(ints) == 1 else ints
        else:
            continue
        ext.add_node_data(node.name, {"debug_handle": dh_val})
    ext.set_sync_key("debug_handle")
    ext.set_label_formatter(lambda d: [f"dh={d.get('debug_handle', '?')}"])
    ext.set_color_rule(NumericColorRule(attribute="debug_handle", cmap="viridis", handle_outliers=False))
    return ext


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

def _write_compare_html(
    output_path: Path,
    ref_payload: Dict[str, Any],
    decomp_payload: Dict[str, Any],
    fused_payload: Dict[str, Any],
) -> None:
    js_bundle = FXGraphExporter._load_viewer_js_bundle()

    payloads_json = json.dumps({
        "ref": ref_payload,
        "decomp": decomp_payload,
        "fused": fused_payload,
    })

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>3-Graph Compare: from_node_root Sync Demo</title>
  <style>
    html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; font-family: sans-serif; background: #f3f4f6; }}
    .topbar {{ min-height: 48px; display: flex; align-items: center; gap: 12px; padding: 8px 14px; border-bottom: 1px solid #d1d5db; background: #ffffff; flex-wrap: wrap; }}
    .title {{ font-weight: 600; font-size: 15px; }}
    .desc {{ font-size: 12px; color: #6b7280; max-width: 900px; }}
    .main {{ height: calc(100% - 56px); padding: 10px; box-sizing: border-box; }}
    #compare_root {{ width: 100%; height: 100%; }}
    .btn {{ padding: 6px 12px; cursor: pointer; border: 1px solid #d1d5db; border-radius: 4px; background: #fff; font-size: 13px; }}
    .btn:hover {{ background: #f0f8ff; }}
    details {{ font-size: 12px; color: #374151; }}
    summary {{ cursor: pointer; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="title">3-Graph Compare: from_node_root Sync Demo</div>
    <button class="btn" id="btn_highlight_demo">Highlight Demo</button>
    <button class="btn" id="btn_clear_highlight">Clear Highlights</button>
    <details>
      <summary>About this demo</summary>
      <p>
        <b>Graph 1 (Reference):</b> float model exported; run_decompositions populates from_node on all nodes.<br>
        <b>Graph 2 (Decomposed, 1→many):</b> each linear → t + mm + add; all 3 share from_node_root="linear".<br>
        <b>Graph 3 (Fused, many→1):</b> deepcopy of ref; relu nodes that follow a linear get a union debug_handle.<br>
        <b>Sync mode 'Auto (from_node→handle→id)'</b>: from_node_root is tried first, then debug_handle set intersection.
      </p>
    </details>
  </div>
  <div class="main">
    <div id="compare_root"></div>
  </div>

  <script>
    const payloads = {payloads_json};
  </script>
  <script>
{js_bundle}
  </script>
  <script>
    window.onload = function() {{
      const ref = FXGraphViewer.create({{
        payload: payloads.ref,
        mount: {{ root: document.createElement('div') }},
        layout: {{ preset: 'split' }},
        state: {{ activeExtensions: ['debug_handle_sync'], colorBy: 'debug_handle_sync' }},
      }});
      ref.init();

      const decomp = FXGraphViewer.create({{
        payload: payloads.decomp,
        mount: {{ root: document.createElement('div') }},
        layout: {{ preset: 'split' }},
        state: {{ activeExtensions: ['debug_handle_sync'], colorBy: 'debug_handle_sync' }},
      }});
      decomp.init();

      const fused = FXGraphViewer.create({{
        payload: payloads.fused,
        mount: {{ root: document.createElement('div') }},
        layout: {{ preset: 'split' }},
        state: {{ activeExtensions: ['debug_handle_sync'], colorBy: 'debug_handle_sync' }},
      }});
      fused.init();

      const compare = FXGraphCompare.create({{
        viewers: new Map([
          ['Reference (float)', ref],
          ['Decomposed (1→many)', decomp],
          ['Fused (many→1)', fused],
        ]),
        layout: {{ container: '#compare_root' }},
        // sync defaults to {{ mode: 'auto' }}
      }});

      window.fxRef = ref;
      window.fxDecomp = decomp;
      window.fxFused = fused;
      window.fxCompare = compare;

      // Demo: highlight all "linear family" nodes across all 3 graphs
      document.getElementById('btn_highlight_demo').onclick = function() {{
        // Collect linear-family node IDs from each graph
        const refLinearIds = ref.store.baseData.nodes
          .filter(n => n.info && (n.info.name || '').includes('linear'))
          .map(n => n.id);
        const refReluIds = ref.store.baseData.nodes
          .filter(n => n.info && (n.info.name || '').includes('relu'))
          .map(n => n.id);
        ref.addHighlightGroup('linear_family', refLinearIds.concat(refReluIds), '#ff6600');

        const decompLinearIds = decomp.store.baseData.nodes
          .filter(n => n.info && (
            (n.info.name || '').includes('t_default') ||
            (n.info.name || '').includes('mm') ||
            (n.info.name || '').includes('add') ||
            (n.info.name || '').includes('relu')
          ))
          .map(n => n.id);
        decomp.addHighlightGroup('linear_family', decompLinearIds, '#ff6600');

        const fusedLinearIds = fused.store.baseData.nodes
          .filter(n => n.info && (
            (n.info.name || '').includes('linear') ||
            (n.info.name || '').includes('relu')
          ))
          .map(n => n.id);
        fused.addHighlightGroup('linear_family', fusedLinearIds, '#ff6600');
      }};

      document.getElementById('btn_clear_highlight').onclick = function() {{
        [ref, decomp, fused].forEach(v => v.clearAllHighlightGroups());
      }};
    }};
  </script>
</body>
</html>
"""
    output_path.write_text(html)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    model = _ToyModel().eval()
    sample = (torch.randn(1, 32),)

    print("Exporting reference graph...")
    ref_ep = torch.export.export(model, sample, strict=False)
    # run_decompositions populates from_node on all nodes
    ref_ep_decomp = ref_ep.run_decompositions({})
    ref_gm = ref_ep_decomp.module()

    print("Building decomposed graph (1-to-many via Transformer)...")
    # _DecomposeLinearPass is a torch.fx.Transformer — it auto-propagates from_node
    decomp_gm = _DecomposeLinearPass(ref_gm).transform()
    # from_node_root is now set on t_default, mm_default, add_tensor → "linear"

    print("Building fused graph (many-to-1)...")
    fused_gm = _build_fused_graph(ref_gm)

    print("Exporting payloads...")
    ref_exp = FXGraphExporter(ref_gm)
    decomp_exp = FXGraphExporter(decomp_gm)
    fused_exp = FXGraphExporter(fused_gm)
    for exp, gm in [(ref_exp, ref_gm), (decomp_exp, decomp_gm), (fused_exp, fused_gm)]:
        exp.add_extension(_build_debug_handle_extension(gm))

    ref_payload = ref_exp.generate_json_payload()
    decomp_payload = decomp_exp.generate_json_payload()
    fused_payload = fused_exp.generate_json_payload()

    output = Path("demo_3graph_compare.html")
    _write_compare_html(output, ref_payload, decomp_payload, fused_payload)
    print(f"Wrote: {output}")
    print()
    print("Expected sync behavior (mode: auto, from_node_root → debug_handle → id):")
    print("  Click linear in Graph 1 → Graph 2: add_tensor (last of {t,mm,add} with from_node_root=linear).")
    print("  Click t_default/mm_default/add_tensor in Graph 2 → Graph 1: linear (from_node_root match).")
    print("  Click relu in Graph 1 → Graph 2: relu_default (from_node_root=relu).")
    print("  Click relu in Graph 3 (union handle) → Graph 1: relu. Graph 2: relu_default.")


if __name__ == "__main__":
    main()
