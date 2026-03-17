#!/usr/bin/env python3
"""Demo for fx_viewer V3 extensions with Swin and Llama models.

This script exports standalone HTML files using the new fx_viewer module and
adds two extension layers:
1) Target/op-type categorical coloring.
2) Topological-order numeric coloring.

Run (from repo root):
  source ~/executorch/.venv/bin/activate
  python examples/demo_fx_viewer_extensions.py --model both
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import torch


from executorch.backends.qualcomm.utils.fx_viewer import (
    CategoricalColorRule,
    FXGraphExporter,
    GraphExtension,
    GraphNode,
    NumericColorRule,
)


def _base_label(node: GraphNode) -> str:
    target = str(node.info.get("target") if node.info.get("op") == "call_function" else node.info.get("op"))
    return target.replace("aten.", "").replace(".default", "")

def _compute_topological_index(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, int]:
    node_ids = [n["id"] for n in nodes]
    indeg = {nid: 0 for nid in node_ids}
    adj: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for e in edges:
        src, dst = e["v"], e["w"]
        if src in adj and dst in indeg:
            adj[src].append(dst)
            indeg[dst] += 1

    q = deque([nid for nid in node_ids if indeg[nid] == 0])
    topo_index: dict[str, int] = {}
    idx = 0

    while q:
        cur = q.popleft()
        topo_index[cur] = idx
        idx += 1
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    # Fallback for unexpected cycles: keep deterministic order.
    for nid in node_ids:
        if nid not in topo_index:
            topo_index[nid] = idx
            idx += 1

    return topo_index


def _build_color_by_type_extension(nodes: list[dict[str, Any]]) -> GraphExtension:
    ext = GraphExtension(id="color_by_type", name="Color By Type")

    for n in nodes:
        info = n.get("info", {})
        ext.add_node_data(
            n["id"],
            {
                "target": str(info.get("target", "unknown")),
                "op": str(info.get("op", "unknown")),
                "color_data": str(info.get("target") if info.get("op") == "call_function" else info.get("op"))

            },
        )

    ext.set_label_formatter(lambda d: [f"color_data: {d.get('color_data', 'unknown')}"])
    ext.set_color_rule(CategoricalColorRule(attribute="color_data"))
    return ext


def _build_topology_extension(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> GraphExtension:
    topo_idx = _compute_topological_index(nodes, edges)

    ext = GraphExtension(id="topological_order", name="Topological Order")
    for n in nodes:
        idx = topo_idx[n["id"]]
        ext.add_node_data(n["id"], {"topo_index": idx})

    ext.set_label_formatter(lambda d: [f"topo: {d.get('topo_index', -1)}"])
    ext.set_tooltip_formatter(
        lambda d: [
            f"Topological index: {d.get('topo_index', -1)}",
        ]
    )
    ext.set_color_rule(NumericColorRule(attribute="topo_index", cmap="viridis", handle_outliers=False))
    return ext


def _export_with_extensions(model: torch.nn.Module, inputs: tuple[Any, ...], output_html: Path) -> None:
    try:
        ep_model = torch.export.export(model, inputs, strict=False)
        ep_model = ep_model.run_decompositions()
        graph_module = ep_model.graph_module
    except Exception:
        graph_module = torch.fx.symbolic_trace(model)

    exporter = FXGraphExporter(graph_module)

    # override base behavior
    exporter.set_base_label_formatter(_base_label)

    base_payload = exporter.generate_json_payload()
    base_nodes = base_payload["base"]["nodes"]
    base_edges = base_payload["base"]["edges"]

    exporter.add_extension(_build_color_by_type_extension(base_nodes))
    exporter.add_extension(_build_topology_extension(base_nodes, base_edges))

    exporter.export_html(str(output_html))


def _build_swin_model() -> tuple[torch.nn.Module, tuple[Any, ...]]:
    from transformers import SwinConfig, SwinForImageClassification

    config = SwinConfig(
        image_size=224,
        patch_size=4,
        num_channels=3,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        num_labels=10,
    )
    model = SwinForImageClassification(config).eval().to("cpu")
    inputs = (torch.rand(1, 3, 224, 224),)
    return model, inputs


def _build_llama_model() -> tuple[torch.nn.Module, tuple[Any, ...]]:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=256,
    )
    model = LlamaForCausalLM(config).eval().to("cpu")
    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    inputs = (input_ids,)
    return model, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="FX Viewer V3 extension demo")
    parser.add_argument(
        "--model",
        choices=["swin", "llama", "both"],
        default="both",
        help="Which model demo to export",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Output directory for generated HTML files",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model in ("swin", "both"):
        print("Building Swin demo model...")
        model, inputs = _build_swin_model()
        out_file = out_dir / "swin_graph_v3_extensions.html"
        _export_with_extensions(model, inputs, out_file)
        print(f"Exported: {out_file}")

    if args.model in ("llama", "both"):
        print("Building Llama demo model...")
        model, inputs = _build_llama_model()
        out_file = out_dir / "llama_graph_v3_extensions.html"
        _export_with_extensions(model, inputs, out_file)
        print(f"Exported: {out_file}")


if __name__ == "__main__":
    main()
