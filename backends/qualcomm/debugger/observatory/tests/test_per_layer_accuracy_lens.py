# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import torch

from executorch.backends.qualcomm.debugger.observatory.interfaces import (
    ObservationContext,
    RecordDigest,
)
from executorch.backends.qualcomm.debugger.observatory.lenses.accuracy import AccuracyLens
from executorch.backends.qualcomm.debugger.observatory.lenses.per_layer_accuracy import (
    PerLayerAccuracyLens,
)


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 4)
        self.fc2 = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)


def _non_io_nodes(graph_module: torch.fx.GraphModule) -> List[torch.fx.Node]:
    return [n for n in graph_module.graph.nodes if n.op not in ("placeholder", "output")]


def _attach_same_root(graph_module: torch.fx.GraphModule, root_name: str) -> None:
    for node in _non_io_nodes(graph_module):
        node.meta["from_node"] = [SimpleNamespace(name=root_name, from_node=[])]


def _attach_ordered_roots(graph_module: torch.fx.GraphModule) -> None:
    for idx, node in enumerate(_non_io_nodes(graph_module)):
        node.meta["from_node"] = [SimpleNamespace(name=f"root_{idx}", from_node=[])]


def test_sparse_index_uses_last_topological_node_per_root() -> None:
    gm = torch.fx.symbolic_trace(_ToyModel().eval())
    _attach_same_root(gm, "shared_root")

    sparse = PerLayerAccuracyLens._build_sparse_node_index(gm)
    key = "root:shared_root"
    assert key in sparse
    assert sparse[key].node_id == _non_io_nodes(gm)[-1].name


def test_sparse_index_fallback_uses_node_id_when_from_node_missing() -> None:
    gm = torch.fx.symbolic_trace(_ToyModel().eval())
    sparse = PerLayerAccuracyLens._build_sparse_node_index(gm)
    node_names = [n.name for n in _non_io_nodes(gm)]
    assert all(f"id:{name}" in sparse for name in node_names)


def test_per_layer_accuracy_observe_analyze_and_frontend_defaults() -> None:
    PerLayerAccuracyLens.clear()

    old_dataset = AccuracyLens._captured_dataset
    old_worst = dict(AccuracyLens._worst_indices)
    try:
        torch.manual_seed(0)
        anchor_model = _ToyModel().eval()
        target_model = _ToyModel().eval()
        with torch.no_grad():
            for p in target_model.parameters():
                p.add_(0.01)

        anchor_gm = torch.fx.symbolic_trace(anchor_model)
        target_gm = torch.fx.symbolic_trace(target_model)
        _attach_ordered_roots(anchor_gm)
        _attach_ordered_roots(target_gm)

        sample = (torch.randn(1, 4),)
        AccuracyLens._captured_dataset = [sample]
        AccuracyLens._worst_indices = {"mse": 0}

        cfg = {
            "per_layer_accuracy": {
                "anchor_record_name": "Exported Float",
                "worst_metric_priority": ["mse"],
            }
        }

        anchor_ctx = ObservationContext(
            config=cfg,
            shared_state={"record_name": "Exported Float"},
        )
        target_ctx = ObservationContext(
            config=cfg,
            shared_state={"record_name": "Quantized Model"},
        )

        anchor_digest = PerLayerAccuracyLens.observe(anchor_gm, anchor_ctx)
        target_digest = PerLayerAccuracyLens.observe(target_gm, target_ctx)

        assert isinstance(anchor_digest, dict)
        assert isinstance(target_digest, dict)
        assert target_digest["sample_index"] == 0
        assert target_digest["match_count"] > 0
        assert len(target_digest["rows"]) > 0
        first_row = target_digest["rows"][0]
        assert "psnr" in first_row
        assert "psnr_drop" not in first_row
        assert "cosine_sim" in first_row
        assert "mse" in first_row
        assert "abs_err" in first_row

        records = [
            RecordDigest(name="Exported Float", timestamp=0.0, data={"per_layer_accuracy": anchor_digest}),
            RecordDigest(name="Quantized Model", timestamp=1.0, data={"per_layer_accuracy": target_digest}),
        ]
        analysis = PerLayerAccuracyLens.analyze(records, {})
        assert "Quantized Model" in analysis.per_record_data
        rec_analysis = analysis.per_record_data["Quantized Model"]
        assert "psnr" in rec_analysis.graph_layers
        psnr_payload = rec_analysis.graph_layers["psnr"].to_payload()
        assert "sparse_match_key" in psnr_payload.sync_keys

        frontend = PerLayerAccuracyLens.get_frontend_spec()
        view = frontend.record(target_digest, {"record": {}}, {"name": "Quantized Model", "index": 1})
        assert view is not None
        graph_blocks = [b for b in view.blocks if getattr(b, "type", "") == "graph"]
        assert graph_blocks
        assert graph_blocks[0].record.default_color_by == "per_layer_accuracy/psnr"
        assert "per_layer_accuracy/psnr" in graph_blocks[0].record.default_layers
        html_blocks = [b for b in view.blocks if getattr(b, "type", "") == "html"]
        assert html_blocks
        assert html_blocks[0].id == "per_layer_accuracy_metrics_table"
        assert "Per-layer Metrics" in html_blocks[0].title
    finally:
        AccuracyLens._captured_dataset = old_dataset
        AccuracyLens._worst_indices = old_worst
        PerLayerAccuracyLens.clear()
