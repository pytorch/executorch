# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from .interfaces import RecordAnalysis


class GraphHub:
    """Framework graph asset/layer registry for report assembly."""

    def __init__(self) -> None:
        self._graph_assets: Dict[str, Dict[str, Any]] = {}
        self._graph_layers: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def register_asset(self, graph_ref: str, base_payload: Dict[str, Any], meta: Dict[str, Any]) -> None:
        if not graph_ref or not isinstance(base_payload, dict):
            return
        self._graph_assets[graph_ref] = {
            "base": base_payload,
            "meta": meta or {},
        }

    def add_analysis_layers(
        self,
        graph_ref: str,
        lens_name: str,
        analysis: RecordAnalysis | None,
    ) -> None:
        """Merge per-record analysis graph layers into hub storage.

        Layer IDs are namespaced internally as `<lens_name>/<layer_key>`.
        """

        if not graph_ref or analysis is None:
            return

        slot = self._graph_layers.setdefault(graph_ref, {})
        for layer_key, contribution in analysis.graph_layers.items():
            if not layer_key.strip():
                continue

            payload = contribution.to_payload()
            namespaced_id = f"{lens_name}/{layer_key}"

            # asdict() forwards every GraphExtensionPayload field (and recurses
            # into per-node GraphExtensionNodePayload) so the JS runtime sees
            # the full schema — including `has_label_formatter`, which gates
            # canvas label rendering and the layer panel's "L" badge.  The
            # original `id` is replaced by `namespaced_id` because the dict
            # key is authoritative on the JS side.
            slot[namespaced_id] = asdict(payload)
            slot[namespaced_id]["id"] = namespaced_id

    def get_asset(self, graph_ref: str) -> Dict[str, Any]:
        return self._graph_assets.get(graph_ref, {})

    def build_payload(self) -> Dict[str, Any]:
        return {
            "graph_assets": self._graph_assets,
            "graph_layers": self._graph_layers,
        }

    @staticmethod
    def build_viewer_payload(graph_assets: Dict[str, Any], graph_layers: Dict[str, Any], graph_ref: str) -> Dict[str, Any]:
        # Layout constants live alongside the JS hardcoded fallback in
        # graph_data_store.js; sourcing them here from FXGraphExporter keeps
        # Python (build-time bbox reservation) and JS (canvas rendering) on
        # the same dimensions so labels never overflow their reserved slot.
        from executorch.devtools.fx_viewer.exporter import FXGraphExporter

        layout = FXGraphExporter._layout_constants_payload()
        asset = graph_assets.get(graph_ref, {})
        if not asset:
            return {
                "base": {"legend": [], "nodes": [], "edges": []},
                "extensions": {},
                "layout": layout,
            }
        return {
            "base": asset.get("base", {"legend": [], "nodes": [], "edges": []}),
            "extensions": graph_layers.get(graph_ref, {}),
            "layout": layout,
        }
