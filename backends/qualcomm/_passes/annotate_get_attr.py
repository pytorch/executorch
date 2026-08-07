# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.qualcomm.builders.node_visitor import dq_ops
from executorch.backends.qualcomm.utils.constants import QCOM_QUANT_ATTRS
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import get_quant_attrs


class AnnotateGetAttr(ExportPass):
    """
    Annotates quantization attributes for `get_attr` nodes.

    In passes such as `CanonicalizeConv`, `ConvertLinearToConv2d`,
    `LiftConstantScalarOperands`, and others, `get_attr` nodes and the
    corresponding quantization attributes are inserted into the GraphModule
    to store modified constant values.

    However, the quantization attributes associated with `get_attr` nodes will
    be discarded in certain passes, such as `I64toI32` and `LayoutTransform`.
    This happens due to the following line:
    `graph_module = super().call(graph_module).graph_module`

    which reconstructs the GraphModule and drops any existing quantization
    attributes stored in the metadata of `get_attr` nodes.

    To guarantee correctness, this pass repopulates the quantization attributes
    for `get_attr` nodes and ensures to be scheduled after the `I64toI32` and
    `LayoutTransform` passes.
    """

    def __init__(self, edge_program: torch.export.ExportedProgram):
        self.edge_program = edge_program

    def _annotate_get_attr(
        self, graph_module: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        for node in graph_module.graph.nodes:
            if node.op == "get_attr" and list(node.users)[0].target in dq_ops:
                dq_op = list(node.users)[0]
                quant_attrs = get_quant_attrs(self.edge_program, dq_op)
                node.meta[QCOM_QUANT_ATTRS] = quant_attrs

    def call(self, graph_module: torch.fx.GraphModule):
        self._annotate_get_attr(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)
