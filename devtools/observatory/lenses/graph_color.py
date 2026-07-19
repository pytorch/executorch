# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graph Color Lens — adds Op Type and Op Target color layers to graph views.

Derives color-by layers from the base graph payload captured by GraphLens.
No runtime observation needed; all work happens in the analyze phase.
"""

from __future__ import annotations

from typing import Any, Dict, List

from executorch.devtools.fx_viewer.color_rules import (
    CategoricalColorRule,
)
from executorch.devtools.fx_viewer.extension import GraphExtension
from executorch.devtools.fx_viewer.models import (
    GraphExtensionNodePayload,
    GraphExtensionPayload,
)

from ..interfaces import (
    AnalysisResult,
    Lens,
    RecordAnalysis,
    RecordDigest,
)

_OP_COLOR_MAP = {
    "call_function": "#4A90D9",
    "placeholder": "#50B86C",
    "output": "#E85D5D",
    "call_module": "#9B6DC6",
    "call_method": "#E8A838",
    "get_attr": "#7190C9",
}


class GraphColorLens(Lens):
    """Produces Op Type and Op Target color layers from graph structure."""

    @classmethod
    def get_name(cls) -> str:
        return "graph_color"

    @staticmethod
    def analyze(
        records: List[RecordDigest], config: Dict[str, Any]
    ) -> AnalysisResult:
        result = AnalysisResult()

        for record in records:
            graph_digest = record.data.get("graph")
            if not graph_digest or not isinstance(graph_digest, dict):
                continue
            base = graph_digest.get("base")
            if not base or not isinstance(base, dict):
                continue
            nodes = base.get("nodes")
            if not nodes:
                continue

            op_type_ext = GraphExtension(id="op_type", name="Op Type")
            target_ext = GraphExtension(id="op_target", name="Op Target")

            for node in nodes:
                node_id = node.get("id", "")
                info = node.get("info", {})
                op = info.get("op", "")
                target_raw = info.get("target", "")

                op_type_ext.add_node_data(node_id, {"op_type": op})

                if op == "call_function":
                    cat = target_raw.replace("aten.", "").replace(".default", "")
                else:
                    cat = op
                target_ext.add_node_data(node_id, {"target_category": cat})

            op_type_ext.set_color_rule(
                CategoricalColorRule("op_type", color_map=_OP_COLOR_MAP)
            )
            target_ext.set_color_rule(CategoricalColorRule("target_category"))

            analysis = RecordAnalysis()
            analysis.add_graph_layer("op_type", op_type_ext)
            analysis.add_graph_layer("op_target", target_ext)
            result.per_record_data[record.name] = analysis

        return result
