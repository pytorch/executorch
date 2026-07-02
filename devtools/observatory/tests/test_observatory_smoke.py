# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import json

from executorch.devtools.observatory import Observatory
from executorch.devtools.observatory.observatory import (
    _NonFiniteFloatAsStringJSONEncoder,
)


class _SmokeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _build_report_payload_for_tests():
    Observatory._ensure_default_lenses()
    return Observatory._generate_report_payload(
        list(Observatory._records.values()),
        Observatory._session_result,
        {},
        Observatory._lens_registry,
    )


def test_observatory_collect_and_export_html(tmp_path) -> None:
    Observatory.clear()

    model = _SmokeModel().eval()
    graph_module = torch.fx.symbolic_trace(model)

    with Observatory.enable_context():
        Observatory.collect("smoke", graph_module)

    out = tmp_path / "report.html"
    Observatory.export_html_report(str(out), title="Smoke")

    assert out.exists()
    assert out.stat().st_size > 0

    Observatory.clear()


def test_observatory_merges_analyze_only_graph_layers_and_defaults() -> None:
    Observatory.clear()


def test_nonfinite_floats_are_serialized_as_strings() -> None:
    payload = {
        "nan_value": float("nan"),
        "pos_inf": float("inf"),
        "neg_inf": float("-inf"),
        "finite": 1.25,
    }

    encoded = json.dumps(payload, cls=_NonFiniteFloatAsStringJSONEncoder)
    decoded = json.loads(encoded)

    assert decoded["nan_value"] == "nan"
    assert decoded["pos_inf"] == "inf"
    assert decoded["neg_inf"] == "-inf"
    assert decoded["finite"] == 1.25

    model = _SmokeModel().eval()
    graph_module = torch.fx.symbolic_trace(model)

    with Observatory.enable_context():
        Observatory.collect("smoke", graph_module)

    payload = _build_report_payload_for_tests()
    assert "smoke" in payload["graph_assets"]

    graph_layers = payload["graph_layers"].get("smoke", {})
    assert "graph_color/op_type" in graph_layers
    assert "graph_color/op_target" in graph_layers

    records = payload["records"]
    assert len(records) == 1
    graph_record = records[0]["views"]["graph"]["blocks"][0]["record"]

    default_layers = graph_record["default_layers"]
    default_color_by = graph_record["default_color_by"]

    assert default_layers == ["graph_color/op_type", "graph_color/op_target"]
    assert default_color_by == "graph_color/op_type"
    for layer_id in default_layers:
        assert layer_id in graph_layers
    assert default_color_by in graph_layers

    Observatory.clear()
