# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm.debugger.observatory.graph_hub import GraphHub
from executorch.backends.qualcomm.debugger.observatory.interfaces import RecordAnalysis
from executorch.backends.qualcomm.utils.fx_viewer import (
    GraphExtensionNodePayload,
    GraphExtensionPayload,
)


def test_graph_hub_register_and_layers() -> None:
    hub = GraphHub()
    hub.register_asset(
        "r0",
        base_payload={"legend": [], "nodes": [{"id": "n0"}], "edges": []},
        meta={"record_name": "r0"},
    )
    analysis = RecordAnalysis()
    analysis.add_graph_layer(
        "error",
        GraphExtensionPayload(
            id="error",
            name="Error",
            legend=[{"label": "L", "color": "#000"}],
            sync_keys=["debug_handle"],
            nodes={
                "n0": GraphExtensionNodePayload(
                    fill_color="#000",
                )
            },
        ),
    )
    hub.add_analysis_layers("r0", "accuracy", analysis)

    payload = hub.build_payload()
    assert "r0" in payload["graph_assets"]
    assert "accuracy/error" in payload["graph_layers"]["r0"]
    assert payload["graph_layers"]["r0"]["accuracy/error"]["sync_keys"] == ["debug_handle"]


def test_build_viewer_payload() -> None:
    graph_assets = {
        "r1": {
            "base": {"legend": [], "nodes": [{"id": "a"}], "edges": []},
            "meta": {},
        }
    }
    graph_layers = {"r1": {"x/y": {"name": "L", "legend": [], "nodes": {}}}}

    payload = GraphHub.build_viewer_payload(graph_assets, graph_layers, "r1")
    assert payload["base"]["nodes"][0]["id"] == "a"
    assert "x/y" in payload["extensions"]
