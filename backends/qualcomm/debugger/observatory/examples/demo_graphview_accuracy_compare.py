#!/usr/bin/env python3
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal Observatory GraphView demo with per-layer accuracy compare support.

Run from repository root:

    source ~/executorch/.venv/bin/activate
    source ~/executorch/qairt/2.37.0.250724/bin/envsetup.sh
    export PYTHONPATH=~/

    python backends/qualcomm/debugger/observatory/examples/demo_graphview_accuracy_compare.py \
        --model toy --output-dir /tmp/observatory_demo

    python backends/qualcomm/debugger/observatory/examples/demo_graphview_accuracy_compare.py \
        --model swin --output-dir /tmp/observatory_demo_swin
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import torch

from executorch.devtools.observatory import Observatory
from executorch.devtools.observatory.interfaces import (
    AnalysisResult,
    Frontend,
    Lens,
    ObservationContext,
    RecordAnalysis,
    RecordDigest,
    TableBlock,
    TableRecordSpec,
    ViewList,
)
from executorch.devtools.fx_viewer import (
    GraphExtensionNodePayload,
    GraphExtensionPayload,
)


@dataclass
class AccuracyGraphArtifact:
    """Artifact wrapper consumed by GraphLens + AccuracyLayerLens."""

    graph_module: torch.fx.GraphModule
    accuracy_layer: Dict[str, Any]


class AccuracyLayerLens(Lens):
    """Demo lens that contributes fx_viewer overlay from per-layer metrics."""

    @classmethod
    def get_name(cls) -> str:
        return "accuracy"

    @classmethod
    def observe(cls, artifact: Any, context: ObservationContext) -> Any:
        if isinstance(artifact, AccuracyGraphArtifact):
            return artifact.accuracy_layer
        return None

    @classmethod
    def digest(cls, observation: Any, context: ObservationContext) -> Any:
        return observation

    @classmethod
    def analyze(cls, records: list[RecordDigest], config: Dict[str, Any]) -> AnalysisResult:
        per_record: Dict[str, RecordAnalysis] = {}
        for record in records:
            digest = record.data.get(cls.get_name())
            if not isinstance(digest, dict):
                continue

            nodes_payload: Dict[str, GraphExtensionNodePayload] = {}
            for node_id, node_value in (digest.get("nodes") or {}).items():
                if not isinstance(node_value, dict):
                    continue
                nodes_payload[node_id] = GraphExtensionNodePayload(
                    info=node_value.get("info") or {},
                    tooltip=node_value.get("tooltip") or [],
                    label_append=node_value.get("label_append") or [],
                    fill_color=node_value.get("fill_color"),
                )

            extension_payload = GraphExtensionPayload(
                id="error",
                name="Accuracy Error",
                legend=digest.get("legend") or [],
                nodes=nodes_payload,
            )

            summary = {
                "nodes_with_metrics": len(nodes_payload),
                "max_mse": digest.get("max_mse", 0.0),
                "mean_mse": digest.get("mean_mse", 0.0),
            }

            record_analysis = RecordAnalysis(data=summary)
            record_analysis.add_graph_layer("error", extension_payload)
            per_record[record.name] = record_analysis

        return AnalysisResult(per_record_data=per_record)

    class AccuracyFrontend(Frontend):
        def record(self, digest, analysis, context):
            if not digest:
                return None

            summary = (analysis or {}).get("record") or {
                "nodes_with_metrics": len((digest.get("nodes") or {}).keys()),
                "max_mse": digest.get("max_mse", 0.0),
                "mean_mse": digest.get("mean_mse", 0.0),
            }
            return ViewList(
                blocks=[
                    TableBlock(
                        id="accuracy_summary",
                        title="Accuracy Summary",
                        record=TableRecordSpec(data=summary),
                        order=20,
                    )
                ]
            )

    @staticmethod
    def get_frontend_spec() -> Frontend:
        return AccuracyLayerLens.AccuracyFrontend()


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


def _patch_swin_window_ops() -> None:
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
    return SwinForImageClassification(config).eval().to("cpu"), (1, 3, 224, 224)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


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
        for p in quantized.parameters():
            p.copy_(_fake_quantize_tensor(p))
    return quantized.eval().to("cpu")


def _trace_graph_module(model: torch.nn.Module) -> torch.fx.GraphModule:
    graph_module = torch.fx.symbolic_trace(model)
    handle = 1
    for node in graph_module.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        node.meta["debug_handle"] = handle
        handle += 1
    return graph_module


def _capture_outputs(
    graph_module: torch.fx.GraphModule, sample_inputs: tuple[torch.Tensor, ...]
) -> Dict[str, Any]:
    class NodeOutputCapturer(torch.fx.Interpreter):
        def __init__(self, module: torch.fx.GraphModule) -> None:
            super().__init__(module)
            self.outputs: Dict[str, Any] = {}

        def run_node(self, n: torch.fx.Node) -> Any:
            result = super().run_node(n)
            if n.op not in ("placeholder", "output"):
                self.outputs[n.name] = result
            return result

    capturer = NodeOutputCapturer(graph_module)
    capturer.run(*sample_inputs)
    return capturer.outputs


def _as_flat_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float64).reshape(-1)
    if isinstance(value, (tuple, list)) and value:
        tensors = [_as_flat_tensor(v) for v in value]
        tensors = [t for t in tensors if t is not None]
        if tensors:
            return torch.cat(tensors)
    return None


def _score_color(value: float, max_value: float) -> str:
    if max_value <= 0.0:
        return "#93c5fd"
    ratio = min(1.0, max(0.0, value / max_value))
    r = int(147 + (185 - 147) * ratio)
    g = int(197 - (197 - 28) * ratio)
    b = int(253 - (253 - 28) * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_accuracy_layer(
    reference_graph: torch.fx.GraphModule,
    candidate_graph: torch.fx.GraphModule,
    sample_inputs: tuple[torch.Tensor, ...],
) -> Dict[str, Any]:
    reference_outputs = _capture_outputs(reference_graph, sample_inputs)
    candidate_outputs = _capture_outputs(candidate_graph, sample_inputs)

    node_scores: Dict[str, float] = {}
    for node_id, candidate_value in candidate_outputs.items():
        if node_id not in reference_outputs:
            continue

        ref_flat = _as_flat_tensor(reference_outputs[node_id])
        cand_flat = _as_flat_tensor(candidate_value)
        if ref_flat is None or cand_flat is None:
            continue

        size = min(ref_flat.numel(), cand_flat.numel())
        if size == 0:
            continue
        ref_flat = ref_flat[:size]
        cand_flat = cand_flat[:size]

        mse = float(torch.mean((cand_flat - ref_flat) ** 2).item())
        node_scores[node_id] = mse

    max_mse = max(node_scores.values()) if node_scores else 0.0
    mean_mse = sum(node_scores.values()) / len(node_scores) if node_scores else 0.0

    nodes = {}
    for node_id, mse in node_scores.items():
        nodes[node_id] = {
            "info": {"mse": mse},
            "label_append": [f"mse={mse:.4e}"],
            "fill_color": _score_color(mse, max_mse),
        }

    return {
        "legend": [
            {"label": "Low Error", "color": "#93c5fd"},
            {"label": "High Error", "color": "#b91c1c"},
        ],
        "nodes": nodes,
        "max_mse": max_mse,
        "mean_mse": mean_mse,
    }


def _empty_accuracy_layer() -> Dict[str, Any]:
    return {
        "legend": [
            {"label": "Low Error", "color": "#93c5fd"},
            {"label": "High Error", "color": "#b91c1c"},
        ],
        "nodes": {},
        "max_mse": 0.0,
        "mean_mse": 0.0,
    }


def _build_model(model_name: str) -> tuple[torch.nn.Module, tuple[int, ...]]:
    if model_name == "toy":
        return _build_toy_model()
    if model_name == "swin":
        return _build_swin_model()
    raise ValueError(f"Unsupported model: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observatory GraphView accuracy compare demo")
    parser.add_argument("--model", choices=["toy", "swin"], default="toy")
    parser.add_argument("--output-dir", default="/tmp/observatory_graphview_demo")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    _set_seed(args.seed)
    model, input_shape = _build_model(args.model)
    sample = (torch.rand(*input_shape),)

    reference_model = model
    candidate_model = _make_fake_quantized_copy(model)

    reference_graph = _trace_graph_module(reference_model)
    candidate_graph = _trace_graph_module(candidate_model)

    accuracy_candidate = _build_accuracy_layer(reference_graph, candidate_graph, sample)
    accuracy_reference = _empty_accuracy_layer()

    Observatory.register_lens(AccuracyLayerLens)
    Observatory.clear()

    with Observatory.enable_context():
        Observatory.collect(
            "Reference Float",
            AccuracyGraphArtifact(graph_module=reference_graph, accuracy_layer=accuracy_reference),
        )
        Observatory.collect(
            "Candidate FakeQuant",
            AccuracyGraphArtifact(graph_module=candidate_graph, accuracy_layer=accuracy_candidate),
        )

    html_path = os.path.join(args.output_dir, f"{args.model}_observatory_report.html")
    json_path = os.path.join(args.output_dir, f"{args.model}_observatory_report.json")

    Observatory.export_html_report(html_path, title=f"Observatory GraphView Demo ({args.model})")
    Observatory.export_json(json_path)

    summary = {
        "report_html": html_path,
        "report_json": json_path,
        "model": args.model,
        "max_mse": accuracy_candidate.get("max_mse", 0.0),
        "mean_mse": accuracy_candidate.get("mean_mse", 0.0),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
