# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch
from executorch.backends.qualcomm.builders.node_visitor import dq_ops, q_ops
from executorch.backends.qualcomm.utils.constants import (
    QCOM_DTYPE,
    QCOM_ENCODING,
    QCOM_QUANT_MAX,
    QCOM_QUANT_MIN,
    QCOM_REQUANTIZE,
    QCOM_SCALE,
    QCOM_ZERO_POINT,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import get_quant_attrs


EDGE_CAT_OPS = {
    exir_ops.edge.aten.cat.default,
    exir_ops.edge.aten.concat.default,
}


class AnnotateConcatRequant(ExportPass):
    """
    Record explicit requantization needs for concat inputs whose concrete
    post-calibration qparams do not match concat's output domain.
    """

    def __init__(
        self,
        edge_program: torch.export.ExportedProgram,
        skip_advanced_requant: bool = False,
    ):
        super(AnnotateConcatRequant, self).__init__()
        self.edge_program = edge_program
        self.skip_advanced_requant = skip_advanced_requant

    def _is_requant_needed(self, src_attrs: Dict[str, Any], dst_attrs: Dict[str, Any]):
        if self.skip_advanced_requant:
            return src_attrs[QCOM_DTYPE] != dst_attrs[QCOM_DTYPE]

        return any(
            src_attrs[attr] != dst_attrs[attr]
            for attr in [
                QCOM_SCALE,
                QCOM_ZERO_POINT,
                QCOM_QUANT_MIN,
                QCOM_QUANT_MAX,
                QCOM_DTYPE,
            ]
        )

    def _annotate_concat_input_requant(self, quant_node: torch.fx.Node) -> None:
        cat_node = quant_node.args[0]
        if cat_node.target not in EDGE_CAT_OPS:
            return

        output_q_attrs = get_quant_attrs(self.edge_program, quant_node)
        for input_node in cat_node.args[0]:
            if input_node.target not in dq_ops:
                continue

            source_q_node = input_node.args[0]
            if source_q_node.target not in q_ops:
                continue

            source_q_attrs = get_quant_attrs(self.edge_program, source_q_node)
            if not self._is_requant_needed(source_q_attrs, output_q_attrs):
                continue

            source_node = source_q_node.args[0]
            if not isinstance(source_node, torch.fx.Node):
                continue

            requant_attrs = output_q_attrs.copy()
            requant_attrs[QCOM_ENCODING] = source_q_attrs[QCOM_ENCODING]
            source_node.meta.setdefault(QCOM_REQUANTIZE, {})
            source_node.meta[QCOM_REQUANTIZE][cat_node.name] = requant_attrs

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if (
                node.target in q_ops
                and isinstance(node.args[0], torch.fx.Node)
                and node.args[0].target in EDGE_CAT_OPS
            ):
                self._annotate_concat_input_requant(node)
        return PassResult(graph_module, True)
