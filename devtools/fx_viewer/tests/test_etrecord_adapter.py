# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the ETRecord -> FXGraphCompareExporter adapter.

Scope: the *new* surfaces only.
  * ``FXGraphCompareExporter`` construction/validation + payload shape.
  * ``build_compare_from_etrecord`` on an in-memory ``ETRecord`` (exercises
    the ``Union[str, ETRecord]`` accept path without touching disk).
  * ``Inspector.export_fx_viewer_html`` delegation contract only.

Non-goals: ``FXGraphExporter`` internals, JS sync behavior, HTML render (the
last would fire the fast-sugiyama layout inside `export_html`; every payload
assertion below goes through ``.generate_json_payload()`` which is a real
layout call but stays in-process). No QNN, no GPU.
"""

from __future__ import annotations

import types
from collections import OrderedDict
from typing import Tuple
from unittest import mock

import pytest
import torch
import torch.nn as nn

from executorch.devtools.etrecord import ETRecord
from executorch.devtools.fx_viewer.compare_exporter import FXGraphCompareExporter
from executorch.devtools.fx_viewer.etrecord_adapter import build_compare_from_etrecord
from executorch.devtools.fx_viewer.exporter import FXGraphExporter
from executorch.exir import to_edge_transform_and_lower


# ---------------------------------------------------------------------------
# Fixtures — tiny conv+relu module, exported once per test module. The whole
# fixture is a few dozen ops, well under a second on CPU.
# ---------------------------------------------------------------------------


class _TinyConvRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(3, 4, 3)
        self.c2 = nn.Conv2d(4, 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))))


@pytest.fixture(scope="module")
def programs() -> Tuple[torch.export.ExportedProgram, torch.export.ExportedProgram]:
    """Return ``(aten_ep, edge_ep)`` for the tiny module. Module-scoped so
    every test in this file shares the same trace — keeps wall time tight."""
    model = _TinyConvRelu().eval()
    inputs = (torch.randn(1, 3, 8, 8),)
    aten_ep = torch.export.export(model, inputs, strict=False)
    edge_ep = to_edge_transform_and_lower(aten_ep).exported_program()
    return aten_ep, edge_ep


@pytest.fixture(scope="module")
def minimal_etrecord(programs) -> ETRecord:
    """Build an ETRecord in memory. Only ``exported_program`` and
    ``edge_dialect_program`` are set — no ``graph_map``, no delegate map.
    ``ETRecord`` warns "please do not construct directly" but the constructor
    is public Python, and this is the cheapest way to exercise the
    ``Union[str, ETRecord]`` branch without round-tripping through disk."""
    aten_ep, edge_ep = programs
    return ETRecord(
        exported_program=aten_ep,
        edge_dialect_program=edge_ep,
    )


def _fresh_exporter(gm: torch.fx.GraphModule) -> FXGraphExporter:
    """Build a new FXGraphExporter — used where several tests each want an
    independent exporter (add_extension mutates state)."""
    return FXGraphExporter(gm)


# ---------------------------------------------------------------------------
# 1. FXGraphCompareExporter payload shape
# ---------------------------------------------------------------------------


def test_compare_exporter_payload_preserves_insertion_order_and_sync_mode(programs):
    aten_ep, edge_ep = programs
    viewers = OrderedDict(
        [
            ("Aten", _fresh_exporter(aten_ep.graph_module)),
            ("Edge", _fresh_exporter(edge_ep.graph_module)),
        ]
    )
    cx = FXGraphCompareExporter(viewers, title="Compare")
    payload = cx.generate_json_payload()

    assert payload["title"] == "Compare"
    assert isinstance(payload["viewers"], list)
    assert [v["name"] for v in payload["viewers"]] == ["Aten", "Edge"]
    for v in payload["viewers"]:
        assert set(v.keys()) == {"name", "payload"}
        # Each viewer payload is a FXGraphExporter payload; sanity: contains base + extensions.
        assert "base" in v["payload"]
        assert "extensions" in v["payload"]
    assert payload["sync"] == {"mode": "auto"}


# ---------------------------------------------------------------------------
# 2. Duplicate viewer names rejected
# ---------------------------------------------------------------------------


def test_compare_exporter_rejects_duplicate_viewer_names(programs):
    aten_ep, _ = programs
    with pytest.raises(ValueError, match="duplicate viewer name"):
        FXGraphCompareExporter(
            [
                ("Same", _fresh_exporter(aten_ep.graph_module)),
                ("Same", _fresh_exporter(aten_ep.graph_module)),
            ]
        )


# ---------------------------------------------------------------------------
# 3. Empty viewer set rejected
# ---------------------------------------------------------------------------


def test_compare_exporter_rejects_empty_viewers():
    with pytest.raises(ValueError, match="at least one viewer"):
        FXGraphCompareExporter([])


# ---------------------------------------------------------------------------
# 4. Adapter on minimal in-memory ETRecord → two panes named Aten + Edge dialect
# ---------------------------------------------------------------------------


def test_build_compare_from_minimal_etrecord_yields_two_named_panes(minimal_etrecord):
    exporter = build_compare_from_etrecord(
        minimal_etrecord,
        title="tiny",
        include_backend_overlay=False,  # no delegate map, keep the code path off
    )
    payload = exporter.generate_json_payload()
    assert payload["title"] == "tiny"
    assert [v["name"] for v in payload["viewers"]] == ["Aten", "Edge dialect"]


# ---------------------------------------------------------------------------
# 5. enrich_missing_meta populates from_node_root on the aten pane
#
# Testing via the observable payload rather than internal ep identity: after
# enrichment, at least one aten-side call node MUST carry ``from_node_root``
# in its ``info`` dict — that's the whole point of the pass and it's what
# fx_viewer's 'auto' sync consumes.
# ---------------------------------------------------------------------------


def test_enrich_missing_meta_populates_from_node_root_on_aten(minimal_etrecord):
    exporter = build_compare_from_etrecord(
        minimal_etrecord,
        enrich_missing_meta=True,
        include_backend_overlay=False,
    )
    payload = exporter.generate_json_payload()
    aten_pane = next(v for v in payload["viewers"] if v["name"] == "Aten")
    aten_nodes = aten_pane["payload"]["base"]["nodes"]
    with_root = [
        n
        for n in aten_nodes
        if isinstance(n.get("info"), dict) and n["info"].get("from_node_root")
    ]
    assert with_root, (
        "expected at least one aten node with from_node_root after enrichment; "
        f"got {len(aten_nodes)} nodes, none carrying from_node_root"
    )


def test_enrich_missing_meta_disabled_leaves_aten_bare(minimal_etrecord):
    # Companion assertion: without enrichment the aten pane emits no
    # from_node_root at all (raw torch.export output has no from_node meta).
    exporter = build_compare_from_etrecord(
        minimal_etrecord,
        enrich_missing_meta=False,
        include_backend_overlay=False,
    )
    payload = exporter.generate_json_payload()
    aten_pane = next(v for v in payload["viewers"] if v["name"] == "Aten")
    for n in aten_pane["payload"]["base"]["nodes"]:
        info = n.get("info") or {}
        assert "from_node_root" not in info or not info["from_node_root"]


# ---------------------------------------------------------------------------
# 6. Backend overlay attaches a `backend` extension covering every edge call node
# ---------------------------------------------------------------------------


def test_include_backend_overlay_attaches_backend_extension(programs):
    aten_ep, edge_ep = programs
    edge_gm = edge_ep.graph_module

    # Every edge call node in the fixture has a scalar debug_handle 1..N.
    handle_map: dict = {}
    for n in edge_gm.graph.nodes:
        if n.op in ("placeholder", "output"):
            continue
        h = n.meta.get("debug_handle")
        if h in (None, 0):
            continue
        handle_map[int(h) if isinstance(h, int) else int(h[0])] = h
    assert handle_map, "fixture invariant: at least one edge node has debug_handle"

    debug_handle_map = {
        "forward": {
            i: [h] if isinstance(h, int) else list(h)
            for i, h in enumerate(handle_map.keys(), start=1)
        }
    }
    delegate_map = {
        "forward": {
            i: {"name": "XnnpackBackend", "delegate_map": {}}
            for i in debug_handle_map["forward"].keys()
        }
    }

    rec = ETRecord(
        exported_program=aten_ep,
        edge_dialect_program=edge_ep,
        _debug_handle_map=debug_handle_map,
        _delegate_map=delegate_map,
    )
    exporter = build_compare_from_etrecord(
        rec,
        enrich_missing_meta=False,
        include_backend_overlay=True,
    )

    # Pre-payload probe: the edge FXGraphExporter carries a `backend` extension.
    edge_viewer = dict(exporter._viewers)["Edge dialect"]
    ext_ids = [ext.id for ext in edge_viewer.extensions]
    assert "backend" in ext_ids, f"expected backend ext on edge pane, got {ext_ids}"
    backend_ext = next(e for e in edge_viewer.extensions if e.id == "backend")
    assert len(backend_ext.nodes_data) > 0
    assert all(
        v.get("backend") == "XnnpackBackend" for v in backend_ext.nodes_data.values()
    )

    # And it survives round-trip through generate_json_payload.
    payload = exporter.generate_json_payload()
    edge_pane = next(v for v in payload["viewers"] if v["name"] == "Edge dialect")
    exts = edge_pane["payload"]["extensions"]  # dict {id: payload}
    assert "backend" in exts
    assert len(exts["backend"]["nodes"]) > 0


def test_include_backend_overlay_false_attaches_no_extension(minimal_etrecord):
    exporter = build_compare_from_etrecord(
        minimal_etrecord,
        enrich_missing_meta=False,
        include_backend_overlay=False,
    )
    edge_viewer = dict(exporter._viewers)["Edge dialect"]
    assert edge_viewer.extensions == []


# ---------------------------------------------------------------------------
# 7. Inspector.export_fx_viewer_html — delegation contract only
#
# We do NOT instantiate Inspector (constructor pulls in etdump + a lot else).
# We call the unbound method against a SimpleNamespace stand-in and patch the
# adapter symbol that Inspector locally imports.
# ---------------------------------------------------------------------------


def test_inspector_export_fx_viewer_html_exists_and_delegates():
    from executorch.devtools.inspector._inspector import Inspector

    # Method is defined on the class.
    assert callable(getattr(Inspector, "export_fx_viewer_html", None))

    fake_self = types.SimpleNamespace(_etrecord=object())
    fake_exporter = mock.MagicMock()

    # Inspector does a local import of build_compare_from_etrecord from the
    # adapter module — patch it at its source.
    with mock.patch(
        "executorch.devtools.fx_viewer.etrecord_adapter.build_compare_from_etrecord",
        return_value=fake_exporter,
    ) as patched:
        Inspector.export_fx_viewer_html(
            fake_self,
            "out.html",
            title="T",
            enrich_missing_meta=False,
            include_backend_overlay=False,
        )

    patched.assert_called_once()
    _, kwargs = patched.call_args
    # First positional is the etrecord we stubbed on fake_self.
    assert patched.call_args.args[0] is fake_self._etrecord
    assert kwargs == {
        "title": "T",
        "enrich_missing_meta": False,
        "include_backend_overlay": False,
    }
    fake_exporter.export_html.assert_called_once_with("out.html")


def test_inspector_export_fx_viewer_html_raises_without_etrecord():
    from executorch.devtools.inspector._inspector import Inspector

    fake_self = types.SimpleNamespace(_etrecord=None)
    with pytest.raises(RuntimeError, match="requires an ETRecord"):
        Inspector.export_fx_viewer_html(fake_self, "out.html")
