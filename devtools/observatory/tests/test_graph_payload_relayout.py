# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from executorch.devtools.observatory.interfaces import (
    AnalysisResult,
    Frontend,
    Lens,
    RecordAnalysis,
    RecordDigest,
    SessionResult,
)
from executorch.devtools.observatory.observatory import Observatory
from executorch.devtools.fx_viewer import (
    FXGraphExporter,
    GraphExtensionNodePayload,
    GraphExtensionPayload,
)


def _node_by_id(base_payload: dict, node_id: str) -> dict:
    for node in base_payload["nodes"]:
        if node.get("id") == node_id:
            return node
    raise KeyError(node_id)


def test_relayout_payload_base_uses_extension_label_lines() -> None:
    base_payload = {
        "legend": [],
        "nodes": [
            {"id": "n0", "label": "a", "x": 0.0, "y": 0.0, "width": 100, "height": 36, "info": {}},
            {"id": "n1", "label": "b", "x": 0.0, "y": 0.0, "width": 100, "height": 36, "info": {}},
        ],
        "edges": [{"v": "n0", "w": "n1", "points": []}],
    }
    ext_payload = {
        "acc/psnr": {
            "name": "PSNR",
            "legend": [],
            "nodes": {
                "n0": {"label_append": ["psnr=12.34567890123456789"]},
            },
        }
    }

    relaid = FXGraphExporter.relayout_payload_base(base_payload, ext_payload)

    assert _node_by_id(base_payload, "n0")["width"] == 100
    assert _node_by_id(base_payload, "n0")["height"] == 36
    assert base_payload["edges"][0]["points"] == []

    n0 = _node_by_id(relaid, "n0")
    assert n0["width"] > 100
    assert n0["height"] > 36
    assert relaid["edges"][0]["points"]


class _FakeGraphLens(Lens):
    @classmethod
    def get_name(cls) -> str:
        return "graph"

    @staticmethod
    def get_frontend_spec() -> Frontend:
        return Frontend()


class _FakeLayerLens(Lens):
    @classmethod
    def get_name(cls) -> str:
        return "fake_layer"

    @staticmethod
    def analyze(records, config) -> AnalysisResult:
        result = AnalysisResult()
        for record in records:
            rec_analysis = RecordAnalysis()
            rec_analysis.add_graph_layer(
                "test",
                GraphExtensionPayload(
                    id="test",
                    name="Test Layer",
                    legend=[],
                    nodes={
                        "n0": GraphExtensionNodePayload(
                            label_append=["a long extension label for relayout"],
                        )
                    },
                ),
            )
            result.per_record_data[record.name] = rec_analysis
        return result

    @staticmethod
    def get_frontend_spec() -> Frontend:
        return Frontend()


def test_observatory_relayout_applied_during_payload_assembly() -> None:
    records = [
        RecordDigest(
            name="r0",
            timestamp=0.0,
            data={
                "graph": {
                    "graph_ref": "r0",
                    "base": {
                        "legend": [],
                        "nodes": [
                            {
                                "id": "n0",
                                "label": "a",
                                "x": 0.0,
                                "y": 0.0,
                                "width": 100,
                                "height": 36,
                                "info": {},
                            },
                            {
                                "id": "n1",
                                "label": "b",
                                "x": 0.0,
                                "y": 0.0,
                                "width": 100,
                                "height": 36,
                                "info": {},
                            },
                        ],
                        "edges": [{"v": "n0", "w": "n1", "points": []}],
                    },
                    "meta": {},
                }
            },
        )
    ]

    payload = Observatory._generate_report_payload(
        records=records,
        session=SessionResult(),
        config={},
        lens_registry=[_FakeGraphLens, _FakeLayerLens],
    )

    base = payload["graph_assets"]["r0"]["base"]
    assert _node_by_id(base, "n0")["width"] > 100
    assert _node_by_id(base, "n0")["height"] > 36
    assert base["edges"][0]["points"]
