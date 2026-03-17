# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Iterable, List


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

    def add_layers(self, graph_ref: str, lens_name: str, layers: Iterable[Dict[str, Any]]) -> None:
        if not graph_ref:
            return
        slot = self._graph_layers.setdefault(graph_ref, {})
        for layer in layers or []:
            layer_id = str(layer.get("id") or "").strip()
            if not layer_id:
                continue
            if "/" not in layer_id:
                layer_id = f"{lens_name}/{layer_id}"
            slot[layer_id] = {
                "name": layer.get("name", layer_id),
                "legend": layer.get("legend", []),
                "nodes": layer.get("nodes", {}),
            }

    def get_asset(self, graph_ref: str) -> Dict[str, Any]:
        return self._graph_assets.get(graph_ref, {})

    def build_payload(self) -> Dict[str, Any]:
        return {
            "graph_assets": self._graph_assets,
            "graph_layers": self._graph_layers,
        }

    @staticmethod
    def build_viewer_payload(graph_assets: Dict[str, Any], graph_layers: Dict[str, Any], graph_ref: str) -> Dict[str, Any]:
        asset = graph_assets.get(graph_ref, {})
        if not asset:
            return {"base": {"legend": [], "nodes": [], "edges": []}, "extensions": {}}
        return {
            "base": asset.get("base", {"legend": [], "nodes": [], "edges": []}),
            "extensions": graph_layers.get(graph_ref, {}),
        }
