#!/usr/bin/env python3
"""Generate unified fx_viewer API harness HTML files.

This generator builds two educational harness outputs:
1) Portable harness (no Qualcomm SDK required):
   - Swin graph
   - real per-layer accuracy extension from fake-quant comparison
   - topology + color-by-type structural extensions
2) Qualcomm harness (requires QNN/QAIRT env):
   - Swin graph
   - real per-layer accuracy extension from Qualcomm PTQ comparison
   - same structural extensions + Qualcomm metadata testcase

Design goals:
1) Few CLI options.
2) One shared HTML template.
3) One shared testcase catalog.
4) Payload/testcase composition done in one place for easy extension.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import torch

from executorch.devtools.fx_viewer import (
    CategoricalColorRule,
    FXGraphExporter,
    GraphExtension,
    NumericColorRule,
)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from harness_testcases import build_testcases


def _load_local_accuracy_demo_module():
    module_path = THIS_DIR / "demo_per_layer_accuracy_fx.py"
    spec = importlib.util.spec_from_file_location("fx_acc_demo_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


acc_demo = _load_local_accuracy_demo_module()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _load_viewer_js_bundle_local() -> str:
    """Load viewer JS from workspace templates.

    We intentionally load from the local repo tree so the harness always uses
    the current in-workspace JS runtime under development.
    """
    template_dir = THIS_DIR.parent / "templates"
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
    chunks: list[str] = []
    for filename in ordered_files:
        path = template_dir / filename
        chunks.append(f"\n// ---- {filename} ----\n")
        chunks.append(path.read_text())
    return "\n".join(chunks)


def _compute_topological_index(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, int]:
    node_ids = [n["id"] for n in nodes]
    indeg = {nid: 0 for nid in node_ids}
    adj: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for e in edges:
        src, dst = e["v"], e["w"]
        if src in adj and dst in indeg:
            adj[src].append(dst)
            indeg[dst] += 1

    q = deque([nid for nid in node_ids if indeg[nid] == 0])
    topo: dict[str, int] = {}
    idx = 0
    while q:
        cur = q.popleft()
        topo[cur] = idx
        idx += 1
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    for nid in node_ids:
        if nid not in topo:
            topo[nid] = idx
            idx += 1

    return topo


def _build_color_by_type_extension(nodes: list[dict[str, Any]]) -> GraphExtension:
    ext = GraphExtension(id="color_by_type", name="Color By Type")
    for n in nodes:
        info = n.get("info", {})
        color_data = str(
            info.get("target") if info.get("op") == "call_function" else info.get("op", "unknown")
        )
        ext.add_node_data(
            n["id"],
            {
                "target": str(info.get("target", "unknown")),
                "op": str(info.get("op", "unknown")),
                "color_data": color_data,
            },
        )

    ext.set_label_formatter(lambda d: [f"color_data: {d.get('color_data', 'unknown')}"])
    ext.set_color_rule(CategoricalColorRule(attribute="color_data"))
    return ext


def _build_topology_extension(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> GraphExtension:
    topo_idx = _compute_topological_index(nodes, edges)
    ext = GraphExtension(id="topological_order", name="Topological Order")
    for n in nodes:
        ext.add_node_data(n["id"], {"topo_index": topo_idx[n["id"]]})

    ext.set_label_formatter(lambda d: [f"topo: {d.get('topo_index', -1)}"])
    ext.set_tooltip_formatter(lambda d: [f"Topological index: {d.get('topo_index', -1)}"])
    ext.set_color_rule(
        NumericColorRule(attribute="topo_index", cmap="viridis", handle_outliers=False)
    )
    return ext


def _add_structural_extensions(exporter: FXGraphExporter) -> None:
    """Attach structural extensions (type + topology) to any exporter."""
    base = exporter.generate_json_payload()
    nodes = base["base"]["nodes"]
    edges = base["base"]["edges"]
    exporter.add_extension(_build_color_by_type_extension(nodes))
    exporter.add_extension(_build_topology_extension(nodes, edges))


def _compute_accuracy_metrics_for_pair(
    graph_pair: Any,
    samples: list[tuple[torch.Tensor, ...]],
) -> tuple[list[Any], int]:
    sample_scores, worst_sample_idx = acc_demo._score_samples_by_e2e_drop(
        graph_pair.reference_graph,
        graph_pair.candidate_graph,
        samples,
    )
    worst_sample = samples[worst_sample_idx]

    reference_outputs = acc_demo._capture_outputs(graph_pair.reference_graph, worst_sample)
    candidate_outputs = acc_demo._capture_outputs(graph_pair.candidate_graph, worst_sample)

    reference_map = acc_demo.get_aot_debug_handle_to_op_name_mapping(graph_pair.reference_graph)
    candidate_map = acc_demo.get_aot_debug_handle_to_op_name_mapping(graph_pair.candidate_graph)
    matches, _ = acc_demo._match_nodes(reference_map, candidate_map)
    metrics = acc_demo._compute_layer_metrics(matches, reference_outputs, candidate_outputs)

    _ = sample_scores  # kept for future extensions/reporting
    return metrics, worst_sample_idx


def _build_swin_samples(num_samples: int) -> tuple[Any, list[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
    model, input_shape = acc_demo._build_swin_model()
    samples = acc_demo._make_random_samples(input_shape, num_samples=num_samples)
    export_sample = samples[0]
    return model, samples, export_sample


def _build_portable_payloads(num_samples: int) -> dict[str, Any]:
    model, samples, export_sample = _build_swin_samples(num_samples)
    graph_pair = acc_demo._build_graph_pair_fake_quant(model, export_sample)
    metrics, worst_sample_idx = _compute_accuracy_metrics_for_pair(graph_pair, samples)

    reference_exporter = FXGraphExporter(graph_pair.reference_graph)
    candidate_exporter = FXGraphExporter(graph_pair.candidate_graph)
    _add_structural_extensions(reference_exporter)
    _add_structural_extensions(candidate_exporter)
    acc_demo._add_accuracy_extension(candidate_exporter, metrics)

    reference_payload = reference_exporter.generate_json_payload()
    candidate_payload = candidate_exporter.generate_json_payload()

    # Second candidate: different fake-quant seed for 3-graph harness demo
    torch.manual_seed(42)
    candidate_model_2 = acc_demo._make_fake_quantized_copy(model)
    candidate_ep_2 = acc_demo._export_with_debug_handles(candidate_model_2, export_sample)
    graph_pair_2 = acc_demo.GraphPair(
        pipeline="fake_quant_2",
        reference_name="Reference Float",
        candidate_name="Candidate Fake-Quantized (seed 42)",
        reference_graph=graph_pair.reference_graph,
        candidate_graph=candidate_ep_2.module(),
        metadata={},
    )
    metrics_2, _ = _compute_accuracy_metrics_for_pair(graph_pair_2, samples)
    candidate_exporter_2 = FXGraphExporter(graph_pair_2.candidate_graph)
    _add_structural_extensions(candidate_exporter_2)
    acc_demo._add_accuracy_extension(candidate_exporter_2, metrics_2)
    candidate_payload_2 = candidate_exporter_2.generate_json_payload()

    return {
        "profile": "portable",
        "model": "swin",
        "method": "fake_quant + intermediate_output_capturer",
        "worst_sample_index": worst_sample_idx,
        "structural": reference_payload,
        "accuracy_reference": reference_payload,
        "accuracy_candidate": candidate_payload,
        "accuracy_candidate_2": candidate_payload_2,
    }


def _build_qualcomm_payloads(
    num_samples: int,
    calibration_steps: int,
    soc_model: str,
    backend: str,
) -> dict[str, Any]:
    model, samples, export_sample = _build_swin_samples(num_samples)
    calibration_samples = samples[: max(1, calibration_steps)]

    graph_pair = acc_demo._build_graph_pair_qualcomm_ptq(
        model=model,
        export_sample=export_sample,
        calibration_samples=calibration_samples,
        soc_model=soc_model,
        backend_name=backend,
    )
    metrics, worst_sample_idx = _compute_accuracy_metrics_for_pair(graph_pair, samples)

    reference_exporter = FXGraphExporter(graph_pair.reference_graph)
    candidate_exporter = FXGraphExporter(graph_pair.candidate_graph)
    _add_structural_extensions(reference_exporter)
    _add_structural_extensions(candidate_exporter)
    acc_demo._add_accuracy_extension(candidate_exporter, metrics)

    reference_payload = reference_exporter.generate_json_payload()
    candidate_payload = candidate_exporter.generate_json_payload()

    return {
        "profile": "qualcomm",
        "model": "swin",
        "method": "QnnQuantizer + prepare_pt2e/convert_pt2e + intermediate_output_capturer",
        "soc_model": soc_model,
        "backend": backend,
        "qnn_sdk_root": os.getenv("QNN_SDK_ROOT", ""),
        "worst_sample_index": worst_sample_idx,
        "structural": reference_payload,
        "accuracy_reference": reference_payload,
        "accuracy_candidate": candidate_payload,
    }


def _load_template() -> str:
    return (THIS_DIR / "harness_template.html").read_text()


def _render_html(payloads: dict[str, Any], testcases: list[dict[str, Any]]) -> str:
    template = _load_template()
    js_bundle = _load_viewer_js_bundle_local()
    payload_json = json.dumps(
        {
            "meta": {
                "profile": payloads.get("profile", "unknown"),
                "model": payloads.get("model", "unknown"),
                "method": payloads.get("method", "unknown"),
                "soc_model": payloads.get("soc_model"),
                "backend": payloads.get("backend"),
                "qnn_sdk_root": payloads.get("qnn_sdk_root"),
                "worst_sample_index": payloads.get("worst_sample_index"),
            },
            "structural": payloads["structural"],
            "accuracy_reference": payloads["accuracy_reference"],
            "accuracy_candidate": payloads["accuracy_candidate"],
            "accuracy_candidate_2": payloads.get("accuracy_candidate_2"),
        }
    )
    testcases_json = json.dumps(testcases)

    out = template.replace("__PAYLOADS_JSON__", payload_json)
    out = out.replace("__TEST_CASES_JSON__", testcases_json)
    out = out.replace("__VIEWER_JS_BUNDLE__", js_bundle)
    return out


def _write_harness(output_path: Path, payloads: dict[str, Any], include_qualcomm: bool) -> None:
    testcases = build_testcases(include_qualcomm=include_qualcomm)
    output_path.write_text(_render_html(payloads, testcases))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unified fx_viewer API test harnesses")
    parser.add_argument(
        "--output-dir",
        default=str(THIS_DIR),
        help="Directory to write generated harness HTML files",
    )
    parser.add_argument("--seed", type=int, default=1126)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--calibration-steps", type=int, default=3)
    parser.add_argument("--soc-model", default="SM8650")
    parser.add_argument("--backend", choices=["htp", "gpu"], default="htp")
    args = parser.parse_args()

    _set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building portable payloads (Swin + fake-quant accuracy)...")
    portable = _build_portable_payloads(num_samples=args.num_samples)

    portable_html = output_dir / "fx_viewer_api_test_harness_portable.html"
    print("[2/4] Writing portable harness...")
    _write_harness(portable_html, portable, include_qualcomm=False)
    print(f"Wrote: {portable_html}")

    qualcomm_html = output_dir / "fx_viewer_api_test_harness_qualcomm.html"
    print("[3/4] Building Qualcomm payloads (if environment is ready)...")
    try:
        qualcomm = _build_qualcomm_payloads(
            num_samples=args.num_samples,
            calibration_steps=args.calibration_steps,
            soc_model=args.soc_model,
            backend=args.backend,
        )
        print("[4/4] Writing Qualcomm harness...")
        _write_harness(qualcomm_html, qualcomm, include_qualcomm=True)
        print(f"Wrote: {qualcomm_html}")
    except Exception as exc:
        message = (
            "Qualcomm harness was not generated because environment setup is incomplete "
            f"or PTQ build failed:\n{exc}\n\n"
            "Portable harness is available and fully functional."
        )
        qualcomm_html.write_text(
            "<html><body><pre style='white-space:pre-wrap;font-family:monospace;'>"
            + message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            + "</pre></body></html>"
        )
        print(f"Wrote fallback note: {qualcomm_html}")


if __name__ == "__main__":
    main()
