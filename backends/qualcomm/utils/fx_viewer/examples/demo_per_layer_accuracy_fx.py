#!/usr/bin/env python3
"""Standalone fx_viewer per-layer accuracy demo (no Observatory UI).

This demo compares two FX graphs and visualizes per-layer accuracy deltas using
an fx_viewer extension. It supports two pipelines:
- fake_quant: backend-agnostic simulated quantization (weight rounding only)
- qualcomm_ptq: Qualcomm PTQ path using QnnQuantizer + prepare/convert PT2E

It also follows the debug workflow:
1) Run end-to-end on multiple input samples.
2) Pick the worst sample by output drop score.
3) Capture per-layer outputs only on that worst sample.

Run from repo root:
  source .venv/bin/activate
  export PYTHONPATH=~/

  # Backend-agnostic demo:
  python backends/qualcomm/utils/fx_viewer/examples/demo_per_layer_accuracy_fx.py \
      --pipeline fake_quant --model swin

  # Qualcomm PTQ demo (requires QNN/QAIRT env):
  source ~/executorch/qairt/2.37.0.250724/bin/envsetup.sh
  python backends/qualcomm/utils/fx_viewer/examples/demo_per_layer_accuracy_fx.py \
      --pipeline qualcomm_ptq --model swin --soc-model SM8650
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch


from executorch.devtools.inspector._inspector_utils import (  # noqa: E402
    DebugHandle,
    get_aot_debug_handle_to_op_name_mapping,
)
from executorch.devtools.inspector._intermediate_output_capturer import (  # noqa: E402
    IntermediateOutputCapturer,
)
from executorch.exir.passes.debug_handle_generator_pass import (  # noqa: E402
    generate_missing_debug_handles,
)

from executorch.backends.qualcomm.utils.fx_viewer import (  # noqa: E402
    FXGraphExporter,
    GraphExtension,
    NumericColorRule,
)


@dataclass
class MatchRecord:
    candidate_node: str
    reference_node: str
    candidate_debug_handle: DebugHandle
    reference_debug_handle: DebugHandle
    matched_by: str


@dataclass
class LayerMetric:
    candidate_node: str
    reference_node: str
    candidate_debug_handle: DebugHandle
    reference_debug_handle: DebugHandle
    matched_by: str
    numel_compared: int
    candidate_shape: str
    reference_shape: str
    max_abs_err: float
    mean_abs_err: float
    mse: float
    cosine_similarity: float
    severity_score: float


@dataclass
class SampleScore:
    sample_index: int
    mse: float
    max_abs_err: float
    cosine_similarity: float
    drop_score: float


@dataclass
class GraphPair:
    pipeline: str
    reference_name: str
    candidate_name: str
    reference_graph: torch.fx.GraphModule
    candidate_graph: torch.fx.GraphModule
    metadata: dict[str, Any]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _patch_swin_window_ops() -> None:
    # Mirrors examples/qualcomm/oss_scripts/swin_transformer.py adjustments.
    from transformers.models.swin import modeling_swin

    def window_partition(input_feature: torch.Tensor, window_size: int) -> torch.Tensor:
        batch_size, height, width, num_channels = input_feature.shape
        input_feature = input_feature.view(
            batch_size,
            height // window_size,
            window_size,
            width // window_size,
            window_size * num_channels,
        )
        windows = input_feature.permute(0, 1, 3, 2, 4).contiguous()
        return windows.view(-1, window_size, window_size, num_channels)

    def window_reverse(
        windows: torch.Tensor, window_size: int, height: int, width: int
    ) -> torch.Tensor:
        num_channels = windows.shape[-1]
        windows = windows.view(
            -1,
            height // window_size,
            width // window_size,
            window_size,
            window_size * num_channels,
        )
        windows = windows.permute(0, 1, 3, 2, 4).contiguous()
        return windows.view(-1, height, width, num_channels)

    modeling_swin.window_partition = window_partition
    modeling_swin.window_reverse = window_reverse


def _build_swin_model() -> tuple[torch.nn.Module, tuple[int, ...]]:
    from transformers import SwinConfig, SwinForImageClassification

    _patch_swin_window_ops()
    config = SwinConfig(
        image_size=224,
        patch_size=4,
        num_channels=3,
        embed_dim=64,
        depths=[1, 1, 1, 1],
        num_heads=[2, 4, 8, 16],
        window_size=7,
        num_labels=10,
    )
    model = SwinForImageClassification(config).eval().to("cpu")
    return model, (1, 3, 224, 224)


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = torch.nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _build_toy_model() -> tuple[torch.nn.Module, tuple[int, ...]]:
    return _ToyModel().eval().to("cpu"), (1, 3, 128, 128)


def _make_random_samples(input_shape: tuple[int, ...], num_samples: int) -> list[tuple[torch.Tensor, ...]]:
    samples: list[tuple[torch.Tensor, ...]] = []
    for _ in range(num_samples):
        samples.append((torch.rand(*input_shape),))
    return samples


def _fake_quantize_tensor(tensor: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor
    qmax = (1 << (num_bits - 1)) - 1
    max_abs = tensor.detach().abs().max()
    if float(max_abs) == 0.0:
        return tensor
    scale = max_abs / float(qmax)
    q = (tensor / scale).round().clamp(-qmax, qmax)
    return q * scale


def _make_fake_quantized_copy(model: torch.nn.Module) -> torch.nn.Module:
    quantized = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in quantized.parameters():
            parameter.copy_(_fake_quantize_tensor(parameter))
    return quantized.eval().to("cpu")


def _export_with_debug_handles(
    model: torch.nn.Module, sample_inputs: tuple[torch.Tensor, ...]
) -> torch.export.ExportedProgram:
    ep = torch.export.export(model, sample_inputs, strict=False)
    generate_missing_debug_handles(ep)
    return ep


def _capture_outputs(
    graph_module: torch.fx.GraphModule, sample_inputs: tuple[torch.Tensor, ...]
) -> Dict[DebugHandle, Any]:
    capturer = IntermediateOutputCapturer(graph_module)
    return capturer.run_and_capture(*sample_inputs)


def _node_to_handle(
    handle_to_nodes: Mapping[DebugHandle, Sequence[str]],
) -> Dict[str, DebugHandle]:
    result: Dict[str, DebugHandle] = {}
    for handle, names in handle_to_nodes.items():
        for name in names:
            result[name] = handle
    return result


def _ensure_graph_module_debug_handles(graph_module: torch.fx.GraphModule) -> None:
    max_handle = 0
    for node in graph_module.graph.nodes:
        handle = node.meta.get("debug_handle")
        if isinstance(handle, int):
            max_handle = max(max_handle, handle)
        elif isinstance(handle, (tuple, list)):
            numeric = [int(x) for x in handle if isinstance(x, int)]
            if numeric:
                max_handle = max(max_handle, max(numeric))

    next_handle = max_handle + 1
    for node in graph_module.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        handle = node.meta.get("debug_handle")
        missing = handle is None or handle == 0 or handle == () or handle == []
        if missing:
            node.meta["debug_handle"] = next_handle
            next_handle += 1


def _match_nodes(
    reference_map: Mapping[DebugHandle, Sequence[str]],
    candidate_map: Mapping[DebugHandle, Sequence[str]],
) -> tuple[list[MatchRecord], dict[str, int]]:
    matches: list[MatchRecord] = []

    ref_node_to_handle = _node_to_handle(reference_map)
    cand_node_to_handle = _node_to_handle(candidate_map)

    matched_candidate_nodes: set[str] = set()

    # Phase 1: exact debug-handle matching.
    for handle in sorted(set(reference_map.keys()) & set(candidate_map.keys())):
        reference_node = reference_map[handle][0]
        for candidate_node in candidate_map[handle]:
            matches.append(
                MatchRecord(
                    candidate_node=candidate_node,
                    reference_node=reference_node,
                    candidate_debug_handle=handle,
                    reference_debug_handle=handle,
                    matched_by="debug_handle",
                )
            )
            matched_candidate_nodes.add(candidate_node)

    # Phase 2: node-name fallback.
    for candidate_node, candidate_handle in cand_node_to_handle.items():
        if candidate_node in matched_candidate_nodes:
            continue
        if candidate_node not in ref_node_to_handle:
            continue
        reference_handle = ref_node_to_handle[candidate_node]
        matches.append(
            MatchRecord(
                candidate_node=candidate_node,
                reference_node=candidate_node,
                candidate_debug_handle=candidate_handle,
                reference_debug_handle=reference_handle,
                matched_by="node_name",
            )
        )
        matched_candidate_nodes.add(candidate_node)

    stats = {
        "reference_handles": len(reference_map),
        "candidate_handles": len(candidate_map),
        "handle_intersection": len(set(reference_map.keys()) & set(candidate_map.keys())),
        "matched_nodes": len(matches),
        "matched_by_debug_handle": sum(1 for m in matches if m.matched_by == "debug_handle"),
        "matched_by_node_name": sum(1 for m in matches if m.matched_by == "node_name"),
        "candidate_nodes_unmatched": max(0, len(cand_node_to_handle) - len(matched_candidate_nodes)),
    }
    return matches, stats


def _flatten_for_metric(value: Any) -> tuple[torch.Tensor | None, str]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float64).reshape(-1), str(tuple(value.shape))

    if isinstance(value, (tuple, list)):
        tensor_parts = [
            v.detach().cpu().to(torch.float64).reshape(-1)
            for v in value
            if isinstance(v, torch.Tensor)
        ]
        if tensor_parts:
            shape = "[" + ", ".join(str(tuple(v.shape)) for v in value if isinstance(v, torch.Tensor)) + "]"
            return torch.cat(tensor_parts), shape
        scalar_parts = [float(v) for v in value if isinstance(v, (int, float))]
        if scalar_parts:
            return torch.tensor(scalar_parts, dtype=torch.float64), f"list(len={len(scalar_parts)})"
        return None, "unsupported_sequence"

    if isinstance(value, (int, float, bool)):
        return torch.tensor([float(value)], dtype=torch.float64), "scalar"

    return None, f"unsupported:{type(value).__name__}"


def _compute_metric_for_pair(
    reference_value: Any,
    candidate_value: Any,
) -> tuple[int, str, str, float, float, float, float] | None:
    ref_flat, ref_shape = _flatten_for_metric(reference_value)
    cand_flat, cand_shape = _flatten_for_metric(candidate_value)

    if ref_flat is None or cand_flat is None:
        return None

    compared = min(ref_flat.numel(), cand_flat.numel())
    if compared == 0:
        return None

    ref = torch.nan_to_num(ref_flat[:compared], nan=0.0, posinf=0.0, neginf=0.0)
    cand = torch.nan_to_num(cand_flat[:compared], nan=0.0, posinf=0.0, neginf=0.0)
    diff = cand - ref
    abs_diff = diff.abs()

    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    mse = float((diff * diff).mean().item())

    ref_norm = float(ref.norm().item())
    cand_norm = float(cand.norm().item())
    if ref_norm == 0.0 or cand_norm == 0.0:
        cosine = 1.0 if ref_norm == cand_norm else 0.0
    else:
        cosine = float(torch.nn.functional.cosine_similarity(ref, cand, dim=0).item())
        if math.isnan(cosine):
            cosine = 0.0

    return compared, ref_shape, cand_shape, max_abs, mean_abs, mse, cosine


def _compute_layer_metrics(
    matches: Iterable[MatchRecord],
    reference_outputs: Mapping[DebugHandle, Any],
    candidate_outputs: Mapping[DebugHandle, Any],
) -> list[LayerMetric]:
    metrics: list[LayerMetric] = []
    for match in matches:
        if match.reference_debug_handle not in reference_outputs:
            continue
        if match.candidate_debug_handle not in candidate_outputs:
            continue
        computed = _compute_metric_for_pair(
            reference_outputs[match.reference_debug_handle],
            candidate_outputs[match.candidate_debug_handle],
        )
        if computed is None:
            continue
        (
            compared,
            reference_shape,
            candidate_shape,
            max_abs,
            mean_abs,
            mse,
            cosine,
        ) = computed
        # Severity of performance drop: larger is worse.
        severity = max_abs + max(0.0, 1.0 - cosine)
        metrics.append(
            LayerMetric(
                candidate_node=match.candidate_node,
                reference_node=match.reference_node,
                candidate_debug_handle=match.candidate_debug_handle,
                reference_debug_handle=match.reference_debug_handle,
                matched_by=match.matched_by,
                numel_compared=compared,
                candidate_shape=candidate_shape,
                reference_shape=reference_shape,
                max_abs_err=max_abs,
                mean_abs_err=mean_abs,
                mse=mse,
                cosine_similarity=cosine,
                severity_score=severity,
            )
        )
    return metrics


def _add_accuracy_extension(exporter: FXGraphExporter, metrics: Iterable[LayerMetric]) -> None:
    ext = GraphExtension(id="per_layer_accuracy", name="Per-layer Accuracy (Worst Sample)")
    for metric in metrics:
        ext.add_node_data(
            metric.candidate_node,
            {
                "reference_node": metric.reference_node,
                "candidate_debug_handle": list(metric.candidate_debug_handle),
                "reference_debug_handle": list(metric.reference_debug_handle),
                "matched_by": metric.matched_by,
                "numel_compared": metric.numel_compared,
                "candidate_shape": metric.candidate_shape,
                "reference_shape": metric.reference_shape,
                "max_abs_err": metric.max_abs_err,
                "mean_abs_err": metric.mean_abs_err,
                "mse": metric.mse,
                "cosine_similarity": metric.cosine_similarity,
                "severity_score": metric.severity_score,
            },
        )

    ext.set_label_formatter(
        lambda d: [
            f"severity={d.get('severity_score', 0.0):.2e}",
            f"max_abs={d.get('max_abs_err', 0.0):.2e}",
        ]
    )
    ext.set_tooltip_formatter(
        lambda d: [
            f"match={d.get('matched_by', 'n/a')}",
            f"ref_node={d.get('reference_node', 'n/a')}",
            f"ref_debug_handle={d.get('reference_debug_handle', [])}",
            f"cand_debug_handle={d.get('candidate_debug_handle', [])}",
            f"shape(ref)={d.get('reference_shape', 'n/a')}",
            f"shape(cand)={d.get('candidate_shape', 'n/a')}",
            f"numel={d.get('numel_compared', 0)}",
            f"severity={d.get('severity_score', 0.0):.6e}",
            f"max_abs={d.get('max_abs_err', 0.0):.6e}",
            f"mean_abs={d.get('mean_abs_err', 0.0):.6e}",
            f"mse={d.get('mse', 0.0):.6e}",
            f"cos={d.get('cosine_similarity', 0.0):.6f}",
        ]
    )
    # Red severity map: higher severity => stronger red.
    ext.set_color_rule(
        NumericColorRule(attribute="severity_score", cmap="reds", handle_outliers=True)
    )
    exporter.add_extension(ext)


def _to_primary_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "logits") and isinstance(value.logits, torch.Tensor):
        return value.logits
    if isinstance(value, (tuple, list)):
        for item in value:
            t = _to_primary_tensor(item)
            if t is not None:
                return t
    if isinstance(value, dict):
        for item in value.values():
            t = _to_primary_tensor(item)
            if t is not None:
                return t
    return None


def _score_samples_by_e2e_drop(
    reference_graph: torch.fx.GraphModule,
    candidate_graph: torch.fx.GraphModule,
    samples: Sequence[tuple[torch.Tensor, ...]],
) -> tuple[list[SampleScore], int]:
    scores: list[SampleScore] = []
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            ref_out = reference_graph(*sample)
            cand_out = candidate_graph(*sample)
            ref_t = _to_primary_tensor(ref_out)
            cand_t = _to_primary_tensor(cand_out)
            if ref_t is None or cand_t is None:
                scores.append(
                    SampleScore(
                        sample_index=idx,
                        mse=float("inf"),
                        max_abs_err=float("inf"),
                        cosine_similarity=0.0,
                        drop_score=float("inf"),
                    )
                )
                continue

            ref = ref_t.detach().cpu().to(torch.float64).reshape(-1)
            cand = cand_t.detach().cpu().to(torch.float64).reshape(-1)
            compared = min(ref.numel(), cand.numel())
            if compared == 0:
                scores.append(
                    SampleScore(
                        sample_index=idx,
                        mse=float("inf"),
                        max_abs_err=float("inf"),
                        cosine_similarity=0.0,
                        drop_score=float("inf"),
                    )
                )
                continue

            ref = torch.nan_to_num(ref[:compared], nan=0.0, posinf=0.0, neginf=0.0)
            cand = torch.nan_to_num(cand[:compared], nan=0.0, posinf=0.0, neginf=0.0)
            diff = cand - ref
            mse = float((diff * diff).mean().item())
            max_abs = float(diff.abs().max().item())

            ref_norm = float(ref.norm().item())
            cand_norm = float(cand.norm().item())
            if ref_norm == 0.0 or cand_norm == 0.0:
                cosine = 1.0 if ref_norm == cand_norm else 0.0
            else:
                cosine = float(torch.nn.functional.cosine_similarity(ref, cand, dim=0).item())
                if math.isnan(cosine):
                    cosine = 0.0

            # Composite score for selecting the worst E2E sample.
            drop_score = max_abs + mse + 5.0 * max(0.0, 1.0 - cosine)
            scores.append(
                SampleScore(
                    sample_index=idx,
                    mse=mse,
                    max_abs_err=max_abs,
                    cosine_similarity=cosine,
                    drop_score=drop_score,
                )
            )

    worst = max(scores, key=lambda s: s.drop_score)
    return scores, worst.sample_index


def _write_compare_html(
    output_path: Path,
    panels: Sequence[dict[str, Any]],
    default_columns: int,
) -> None:
    js_bundle = FXGraphExporter._load_viewer_js_bundle()

    panel_html = []
    for idx, panel in enumerate(panels):
        panel_html.append(
            f"""
    <div class=\"panel\">
      <div class=\"panel-header\">{panel['title']}</div>
      <div id=\"viewer_{idx}\" class=\"viewer\"></div>
    </div>
"""
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset=\"UTF-8\">
  <title>FX Viewer Accuracy Compare</title>
  <style>
    :root {{ --cols: {max(1, default_columns)}; }}
    html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; font-family: sans-serif; background: #f3f4f6; }}
    .topbar {{ min-height: 48px; display: flex; align-items: center; gap: 12px; padding: 8px 14px; border-bottom: 1px solid #d1d5db; background: #ffffff; flex-wrap: wrap; }}
    .title {{ font-weight: 600; }}
    .main {{ display: grid; grid-template-columns: repeat(var(--cols), minmax(420px, 1fr)); gap: 10px; height: calc(100% - 64px); padding: 10px; box-sizing: border-box; overflow: auto; }}
    .panel {{ background: #ffffff; border: 1px solid #d1d5db; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; min-height: 500px; }}
    .panel-header {{ padding: 8px 10px; font-size: 13px; font-weight: 600; border-bottom: 1px solid #e5e7eb; }}
    .viewer {{ flex: 1; min-height: 0; }}
    label {{ font-size: 13px; }}
    select {{ font-size: 13px; }}
  </style>
</head>
<body>
  <div class=\"topbar\">
    <div class=\"title\">FX Graph Multi-Compare</div>
    <label><input id=\"syncToggle\" type=\"checkbox\" checked /> Sync selection</label>
    <label><input id=\"compactToggle\" type=\"checkbox\" checked /> Compact split mode (hide sidebars)</label>
    <label>Columns:
      <select id=\"columnSelect\">
        <option value=\"1\">1</option>
        <option value=\"2\">2</option>
        <option value=\"3\">3</option>
        <option value=\"4\">4</option>
      </select>
    </label>
  </div>
  <div class=\"main\" id=\"mainGrid\">{''.join(panel_html)}
  </div>
  <script>
    const panels = {json.dumps(panels)};
  </script>
  <script>
{js_bundle}
  </script>
  <script>
    const viewers = [];
    for (let i = 0; i < panels.length; i++) {{
      const containerId = `viewer_${{i}}`;
      const viewer = FXGraphViewer.create({{
        payload: panels[i].payload,
        mount: {{ root: `#${{containerId}}` }},
        layout: {{ preset: 'split' }},
      }});
      viewer.init();

      if (viewer.store.extensions['per_layer_accuracy']) {{
        viewer.setLayers(Object.keys(viewer.store.extensions));
        viewer.setColorBy('per_layer_accuracy');
      }}
      viewers.push(viewer);
      window[`fxViewer_${{i}}`] = viewer;
    }}

    const syncToggle = document.getElementById('syncToggle');
    const compactToggle = document.getElementById('compactToggle');
    const columnSelect = document.getElementById('columnSelect');
    const mainGrid = document.getElementById('mainGrid');
    columnSelect.value = String(Math.max(1, Math.min(4, {max(1, default_columns)})));

    function applyColumns() {{
      const cols = Number(columnSelect.value || 2);
      document.documentElement.style.setProperty('--cols', String(cols));
    }}

    const compare = FXGraphCompare.create({{
      viewers,
      layout: {{ columns: Number(columnSelect.value || 2), compact: true }},
      sync: {{ selection: true }},
    }});

    function setCompactMode(enabled) {{
      compare.setCompact(enabled);
    }}

    compactToggle.addEventListener('change', () => setCompactMode(compactToggle.checked));
    syncToggle.addEventListener('change', () => compare.setSync({{ selection: syncToggle.checked }}));
    columnSelect.addEventListener('change', applyColumns);

    applyColumns();
    setCompactMode(true);
  </script>
</body>
</html>
"""
    output_path.write_text(html)


def _write_metrics_json(
    output_path: Path,
    metrics: Sequence[LayerMetric],
    match_stats: Mapping[str, int],
    metadata: Mapping[str, Any],
    sample_scores: Sequence[SampleScore],
    worst_sample_index: int,
) -> None:
    payload = {
        "metadata": dict(metadata),
        "match_stats": dict(match_stats),
        "worst_sample_index": worst_sample_index,
        "sample_scores": [asdict(s) for s in sample_scores],
        "summary": {
            "layers_with_metrics": len(metrics),
            "severity_max": max((m.severity_score for m in metrics), default=0.0),
            "severity_mean": (
                sum(m.severity_score for m in metrics) / len(metrics) if metrics else 0.0
            ),
            "max_abs_err_max": max((m.max_abs_err for m in metrics), default=0.0),
            "max_abs_err_mean": (
                sum(m.max_abs_err for m in metrics) / len(metrics) if metrics else 0.0
            ),
            "cosine_similarity_mean": (
                sum(m.cosine_similarity for m in metrics) / len(metrics) if metrics else 0.0
            ),
        },
        "layers": [asdict(metric) for metric in metrics],
        "top10_severity": [
            asdict(metric)
            for metric in sorted(metrics, key=lambda m: m.severity_score, reverse=True)[:10]
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2))


def _build_graph_pair_fake_quant(
    model: torch.nn.Module,
    export_sample: tuple[torch.Tensor, ...],
) -> GraphPair:
    reference_ep = _export_with_debug_handles(model, export_sample)
    candidate_model = _make_fake_quantized_copy(model)
    candidate_ep = _export_with_debug_handles(candidate_model, export_sample)
    return GraphPair(
        pipeline="fake_quant",
        reference_name="Reference Float",
        candidate_name="Candidate Fake-Quantized",
        reference_graph=reference_ep.module(),
        candidate_graph=candidate_ep.module(),
        metadata={
            "method": "deepcopy + weight rounding to int8 grid",
            "qnn_sdk_root": os.getenv("QNN_SDK_ROOT", ""),
        },
    )


def _build_graph_pair_qualcomm_ptq(
    model: torch.nn.Module,
    export_sample: tuple[torch.Tensor, ...],
    calibration_samples: Sequence[tuple[torch.Tensor, ...]],
    soc_model: str,
    backend_name: str,
) -> GraphPair:
    from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QnnExecuTorchBackendType,
    )
    try:
        from executorch.examples.qualcomm.utils import make_quantizer
    except ModuleNotFoundError:
        from examples.qualcomm.utils import make_quantizer
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    if os.getenv("QNN_SDK_ROOT") is None:
        raise RuntimeError(
            "QNN_SDK_ROOT is not set. Run: source ~/executorch/qairt/2.37.0.250724/bin/envsetup.sh"
        )

    backend = getattr(QnnExecuTorchBackendType, f"k{backend_name.title()}Backend")

    reference_ep = _export_with_debug_handles(model, export_sample)
    quant_input_graph = reference_ep.module()
    reference_graph = copy.deepcopy(quant_input_graph)

    quantizer = make_quantizer(
        quant_dtype=QuantDtype.use_8a8w,
        backend=backend,
        soc_model=soc_model,
    )
    annotated_model = prepare_pt2e(quant_input_graph, quantizer)

    with torch.no_grad():
        for sample in calibration_samples:
            annotated_model(*sample)

    candidate_graph = convert_pt2e(annotated_model)
    _ensure_graph_module_debug_handles(candidate_graph)

    return GraphPair(
        pipeline="qualcomm_ptq",
        reference_name="Reference Float (Exported)",
        candidate_name=f"Candidate Qualcomm PTQ ({soc_model}, {backend_name.upper()})",
        reference_graph=reference_graph,
        candidate_graph=candidate_graph,
        metadata={
            "method": "QnnQuantizer + prepare_pt2e/convert_pt2e",
            "soc_model": soc_model,
            "backend": backend_name,
            "calibration_samples": len(calibration_samples),
            "qnn_sdk_root": os.getenv("QNN_SDK_ROOT", ""),
        },
    )


def _run_single_pipeline(
    graph_pair: GraphPair,
    samples: Sequence[tuple[torch.Tensor, ...]],
    pipeline_output_dir: Path,
    seed: int,
    model_name: str,
    default_compare_columns: int,
) -> None:
    pipeline_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{graph_pair.pipeline}] Scoring end-to-end drop over {len(samples)} samples...")
    sample_scores, worst_sample_idx = _score_samples_by_e2e_drop(
        graph_pair.reference_graph,
        graph_pair.candidate_graph,
        samples,
    )
    worst_sample = samples[worst_sample_idx]
    worst_score = next(s for s in sample_scores if s.sample_index == worst_sample_idx)
    print(
        f"[{graph_pair.pipeline}] Worst sample idx={worst_sample_idx}, "
        f"drop={worst_score.drop_score:.6e}, max_abs={worst_score.max_abs_err:.6e}, "
        f"mse={worst_score.mse:.6e}, cos={worst_score.cosine_similarity:.6f}"
    )

    print(f"[{graph_pair.pipeline}] Capturing per-layer outputs on worst sample...")
    reference_outputs = _capture_outputs(graph_pair.reference_graph, worst_sample)
    candidate_outputs = _capture_outputs(graph_pair.candidate_graph, worst_sample)

    print(f"[{graph_pair.pipeline}] Building debug-handle mappings...")
    reference_map = get_aot_debug_handle_to_op_name_mapping(graph_pair.reference_graph)
    candidate_map = get_aot_debug_handle_to_op_name_mapping(graph_pair.candidate_graph)
    matches, match_stats = _match_nodes(reference_map, candidate_map)

    print(f"[{graph_pair.pipeline}] Computing per-layer metrics...")
    metrics = _compute_layer_metrics(matches, reference_outputs, candidate_outputs)

    print(f"[{graph_pair.pipeline}] Exporting fx_viewer HTML...")
    reference_exporter = FXGraphExporter(graph_pair.reference_graph)
    candidate_exporter = FXGraphExporter(graph_pair.candidate_graph)
    _add_accuracy_extension(candidate_exporter, metrics)

    reference_html = pipeline_output_dir / "reference_fx_graph.html"
    candidate_html = pipeline_output_dir / "candidate_fx_graph_per_layer_accuracy.html"
    compare_html = pipeline_output_dir / "compare_side_by_side.html"
    metrics_json = pipeline_output_dir / "per_layer_accuracy_metrics.json"

    reference_payload = reference_exporter.generate_json_payload()
    candidate_payload = candidate_exporter.generate_json_payload()

    reference_exporter.export_html(str(reference_html))
    candidate_exporter.export_html(str(candidate_html))

    panels = [
        {
            "title": graph_pair.reference_name,
            "payload": reference_payload,
        },
        {
            "title": f"{graph_pair.candidate_name} [worst sample={worst_sample_idx}]",
            "payload": candidate_payload,
        },
    ]
    _write_compare_html(compare_html, panels=panels, default_columns=default_compare_columns)

    _write_metrics_json(
        metrics_json,
        metrics,
        match_stats=match_stats,
        metadata={
            "pipeline": graph_pair.pipeline,
            "model": model_name,
            "seed": seed,
            "reference_node_count": len(list(graph_pair.reference_graph.graph.nodes)),
            "candidate_node_count": len(list(graph_pair.candidate_graph.graph.nodes)),
            "reference_captured_outputs": len(reference_outputs),
            "candidate_captured_outputs": len(candidate_outputs),
            **graph_pair.metadata,
        },
        sample_scores=sample_scores,
        worst_sample_index=worst_sample_idx,
    )

    top5 = sorted(metrics, key=lambda m: m.severity_score, reverse=True)[:5]
    print(f"[{graph_pair.pipeline}] Done. Output: {pipeline_output_dir}")
    print(f"[{graph_pair.pipeline}] Top-5 severity layers (red = worse):")
    for item in top5:
        print(
            "  "
            f"{item.candidate_node}: severity={item.severity_score:.6e}, "
            f"max_abs={item.max_abs_err:.6e}, cos={item.cosine_similarity:.6f}, "
            f"match={item.matched_by}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone per-layer accuracy demo for fx_viewer.")
    parser.add_argument("--model", choices=["toy", "swin"], default="swin")
    parser.add_argument("--output-dir", default="fx_viewer_accuracy_demo")
    parser.add_argument("--seed", type=int, default=1126)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--pipeline",
        choices=["fake_quant", "qualcomm_ptq", "both"],
        default="both",
        help="Which comparison pipeline(s) to run.",
    )
    parser.add_argument("--soc-model", default="SM8650")
    parser.add_argument("--backend", choices=["htp", "gpu"], default="htp")
    parser.add_argument("--calibration-steps", type=int, default=4)
    parser.add_argument("--compare-columns", type=int, default=2)
    args = parser.parse_args()

    _set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "swin":
        reference_model, input_shape = _build_swin_model()
    else:
        reference_model, input_shape = _build_toy_model()

    samples = _make_random_samples(input_shape, max(1, args.num_samples))
    export_sample = samples[0]

    requested_pipelines = (
        ["fake_quant", "qualcomm_ptq"] if args.pipeline == "both" else [args.pipeline]
    )

    for pipeline in requested_pipelines:
        if pipeline == "fake_quant":
            pair = _build_graph_pair_fake_quant(reference_model, export_sample)
        elif pipeline == "qualcomm_ptq":
            calib_samples = samples[: max(1, args.calibration_steps)]
            pair = _build_graph_pair_qualcomm_ptq(
                reference_model,
                export_sample,
                calibration_samples=calib_samples,
                soc_model=args.soc_model,
                backend_name=args.backend,
            )
        else:
            raise AssertionError(f"Unsupported pipeline: {pipeline}")

        _run_single_pipeline(
            pair,
            samples=samples,
            pipeline_output_dir=output_dir / pipeline,
            seed=args.seed,
            model_name=args.model,
            default_compare_columns=max(1, min(4, args.compare_columns)),
        )

    print("\nDemo complete.")
    print(f"Output root: {output_dir}")
    for pipeline in requested_pipelines:
        print(f" - {pipeline}/reference_fx_graph.html")
        print(f" - {pipeline}/candidate_fx_graph_per_layer_accuracy.html")
        print(f" - {pipeline}/compare_side_by_side.html")
        print(f" - {pipeline}/per_layer_accuracy_metrics.json")


if __name__ == "__main__":
    main()
