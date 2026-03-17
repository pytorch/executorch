# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm.debugger.observatory.graph_hub import GraphHub


def test_graph_hub_register_and_layers() -> None:
    hub = GraphHub()
    hub.register_asset(
        "r0",
        base_payload={"legend": [], "nodes": [{"id": "n0"}], "edges": []},
        meta={"record_name": "r0"},
    )
    hub.add_layers(
        "r0",
        "accuracy",
        [
            {
                "id": "error",
                "name": "Error",
                "legend": [{"label": "L", "color": "#000"}],
                "nodes": {"n0": {"fill_color": "#000"}},
            }
        ],
    )

    payload = hub.build_payload()
    assert "r0" in payload["graph_assets"]
    assert "accuracy/error" in payload["graph_layers"]["r0"]


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
