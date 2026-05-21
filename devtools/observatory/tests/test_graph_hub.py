# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from executorch.devtools.observatory.graph_hub import GraphHub
from executorch.devtools.observatory.interfaces import (
    GraphLayerContribution,
    RecordAnalysis,
)
from executorch.devtools.fx_viewer import (
    GraphExtension,
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


# ---------------------------------------------------------------------------
# Python ↔ JS extension-payload schema contract
#
# JS GraphDataStore reads `name`, `legend`, `nodes`, `sync_keys`, and
# `has_label_formatter` off `payload.extensions[extId]`. The JS LRU and the
# layer panel's "L" badge depend on `has_label_formatter`, so a silent drop
# of any of these fields breaks node-label rendering on canvas.
# ---------------------------------------------------------------------------

# Fields that GraphHub MUST forward verbatim from a GraphExtensionPayload into
# the per-layer dict that lands in `payload.extensions[extId]` for the JS
# runtime.  `id` is intentionally excluded — it lives in the dict key.
_REQUIRED_EXTENSION_DICT_FIELDS = frozenset(
    f.name for f in fields(GraphExtensionPayload) if f.name != "id"
)


def _make_label_extension(
    ext_id: str = "m", *, with_label: bool = True
) -> GraphExtension:
    ext = GraphExtension(id=ext_id, name="Metric")
    ext.add_node_data("n0", {"foo": 1.0})
    if with_label:
        ext.set_label_formatter(lambda d: [f"foo={d.get('foo', 0):.2f}"])
    return ext


def test_graph_hub_preserves_has_label_formatter() -> None:
    """JS reads `has_label_formatter` to gate label rendering and the L badge."""
    hub = GraphHub()
    hub.register_asset("r0", {"legend": [], "nodes": [{"id": "n0"}], "edges": []}, {})
    analysis = RecordAnalysis()
    analysis.add_graph_layer("m", _make_label_extension())

    hub.add_analysis_layers("r0", "lens", analysis)

    slot = hub.build_payload()["graph_layers"]["r0"]["lens/m"]
    assert slot.get("has_label_formatter") is True


def test_graph_hub_no_label_formatter_emits_false() -> None:
    """Layer without a label formatter must serialize an explicit `False`."""
    hub = GraphHub()
    hub.register_asset("r0", {"legend": [], "nodes": [{"id": "n0"}], "edges": []}, {})
    analysis = RecordAnalysis()
    analysis.add_graph_layer("m", _make_label_extension(with_label=False))

    hub.add_analysis_layers("r0", "lens", analysis)

    slot = hub.build_payload()["graph_layers"]["r0"]["lens/m"]
    assert slot.get("has_label_formatter") is False


def test_graph_hub_layer_dict_has_full_extension_contract() -> None:
    """All non-id GraphExtensionPayload fields must round-trip through GraphHub.

    Adding a new field to GraphExtensionPayload without forwarding it through
    GraphHub silently breaks the JS contract.  This test pins the contract.
    """
    hub = GraphHub()
    hub.register_asset("r0", {"legend": [], "nodes": [{"id": "n0"}], "edges": []}, {})
    payload = GraphExtensionPayload(
        id="m",
        name="Metric",
        legend=[{"label": "L", "color": "#000"}],
        sync_keys=["debug_handle"],
        has_label_formatter=True,
        nodes={"n0": GraphExtensionNodePayload(label_append=["x=1"])},
    )
    analysis = RecordAnalysis()
    analysis.add_graph_layer("m", payload)
    hub.add_analysis_layers("r0", "lens", analysis)

    slot = hub.build_payload()["graph_layers"]["r0"]["lens/m"]
    missing = _REQUIRED_EXTENSION_DICT_FIELDS - set(slot)
    assert not missing, f"GraphHub dropped fields: {sorted(missing)}"


def test_graph_hub_node_payload_dict_has_full_node_contract() -> None:
    """Per-node extension entries must carry every GraphExtensionNodePayload field."""
    hub = GraphHub()
    hub.register_asset("r0", {"legend": [], "nodes": [{"id": "n0"}], "edges": []}, {})
    payload = GraphExtensionPayload(
        id="m",
        name="Metric",
        nodes={
            "n0": GraphExtensionNodePayload(
                info={"k": 1},
                tooltip=["t"],
                label_append=["lbl"],
                fill_color="#abc",
            )
        },
    )
    analysis = RecordAnalysis()
    analysis.add_graph_layer("m", payload)
    hub.add_analysis_layers("r0", "lens", analysis)

    node_dict = hub.build_payload()["graph_layers"]["r0"]["lens/m"]["nodes"]["n0"]
    expected_fields = {f.name for f in fields(GraphExtensionNodePayload)}
    missing = expected_fields - set(node_dict)
    assert not missing, f"node payload dropped fields: {sorted(missing)}"


def test_to_payload_with_overrides_preserves_has_label_formatter() -> None:
    """`id_override`/`name_override` must not strip the rest of the contract.

    The override branch in interfaces.GraphLayerContribution.to_payload re-
    constructs a GraphExtensionPayload; if it omits a field, JS loses it.
    """
    contribution = GraphLayerContribution(
        extension=_make_label_extension(),
        id_override="renamed_id",
        name_override="Renamed",
    )
    payload = contribution.to_payload()

    assert payload.id == "renamed_id"
    assert payload.name == "Renamed"
    assert payload.has_label_formatter is True


def test_to_payload_with_overrides_preserves_sync_keys() -> None:
    payload_in = GraphExtensionPayload(
        id="m",
        name="Metric",
        sync_keys=["debug_handle", "from_node"],
        has_label_formatter=True,
        nodes={"n0": GraphExtensionNodePayload(label_append=["x"])},
    )
    contribution = GraphLayerContribution(
        extension=payload_in,
        id_override="renamed",
    )
    payload_out = contribution.to_payload()

    assert payload_out.sync_keys == ["debug_handle", "from_node"]
    assert payload_out.has_label_formatter is True


def test_graph_hub_preserves_has_label_formatter_through_overrides() -> None:
    """End-to-end: lens contributes with overrides → JS-bound dict still flagged."""
    hub = GraphHub()
    hub.register_asset("r0", {"legend": [], "nodes": [{"id": "n0"}], "edges": []}, {})
    analysis = RecordAnalysis()
    analysis.add_graph_layer(
        "m",
        _make_label_extension(),
        id_override="metric_v2",
        name_override="Metric V2",
    )

    hub.add_analysis_layers("r0", "lens", analysis)

    slot = hub.build_payload()["graph_layers"]["r0"]["lens/m"]
    assert slot.get("name") == "Metric V2"
    assert slot.get("has_label_formatter") is True


def test_build_viewer_payload_includes_layout_constants() -> None:
    """JS reads `payload.layout` for line_height / max_label_extensions / etc.

    Without this, JS silently uses the fallback constants in graph_data_store.js;
    those happen to match today's Python defaults but will drift on first
    Python-side change to `_NODE_LINE_HEIGHT` or `_MAX_LABEL_EXTENSIONS`.
    """
    from executorch.devtools.fx_viewer.exporter import FXGraphExporter

    graph_assets = {
        "r1": {"base": {"legend": [], "nodes": [{"id": "a"}], "edges": []}, "meta": {}}
    }
    payload = GraphHub.build_viewer_payload(graph_assets, {"r1": {}}, "r1")

    assert "layout" in payload
    expected = FXGraphExporter._layout_constants_payload()
    for key in ("line_height", "y_padding", "max_label_extensions", "base_font_px"):
        assert payload["layout"][key] == expected[key]


def test_build_viewer_payload_layout_present_for_missing_graph_ref() -> None:
    """Empty asset path must still emit layout — JS fallback path shouldn't kick in."""
    payload = GraphHub.build_viewer_payload({}, {}, "missing")
    assert "layout" in payload
    assert "max_label_extensions" in payload["layout"]
