# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from executorch.devtools.fx_viewer import FXGraphExporter

from ..interfaces import Frontend, GraphView, Lens, ObservationContext, ViewList


class GraphLens(Lens):
    """Canonical producer of base fx_viewer graph payload per record."""

    @classmethod
    def get_name(cls) -> str:
        return "graph"

    @classmethod
    def _to_graph_module(cls, artifact: Any) -> Optional[torch.fx.GraphModule]:
        if isinstance(artifact, torch.fx.GraphModule):
            return artifact

        graph_module = getattr(artifact, "graph_module", None)
        if isinstance(graph_module, torch.fx.GraphModule):
            return graph_module

        try:
            from torch.export import ExportedProgram

            if isinstance(artifact, ExportedProgram):
                return artifact.graph_module

            exported_program = getattr(artifact, "exported_program", None)
            if isinstance(exported_program, ExportedProgram):
                return exported_program.graph_module
        except Exception:
            pass

        return None

    @classmethod
    def observe(cls, artifact: Any, context: ObservationContext) -> Any:

        graph_config = context.config.get("graph", {})
        if not graph_config.get("enabled", True):
            return None

        graph_module = cls._to_graph_module(artifact)
        if graph_module is None:
            return None

        exporter = FXGraphExporter(graph_module)
        payload = exporter.generate_json_payload()

        base = payload.get("base", {})
        record_name = str(context.shared_state.get("record_name") or "record")

        return {
            "graph_ref": record_name,
            "base": base,
            "meta": {
                "record_name": record_name,
                "node_count": len(base.get("nodes", [])),
                "edge_count": len(base.get("edges", [])),
            },
        }

    @classmethod
    def digest(cls, observation: Any, context: ObservationContext) -> Any:
        return observation

    class GraphFrontend(Frontend):
        def record(self, digest, analysis, context) -> Optional[ViewList]:
            if not digest:
                return None

            view = GraphView(
                id="graph_main",
                title="Graph",
                graph_ref=str(digest.get("graph_ref", "")),
                default_layers=["graph_color/op_type", "graph_color/op_target"],
                default_color_by="graph_color/op_type",
                order=10,
            )
            return ViewList(blocks=[view.as_block()])

    @staticmethod
    def get_frontend_spec() -> Frontend:
        return GraphLens.GraphFrontend()
