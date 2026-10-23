# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import torch

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpElementWiseMod, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Remainder(NodeVisitor):
    """Integer remainder as QNN Mod, whose result takes the divisor's sign like torch.

    Float remainder is decomposed by DecomposeRemainder; Mod has no float support.
    """

    target = ["aten.remainder.Tensor"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        out_tensor = self.get_tensor(node, node)
        if out_tensor.is_floating_point():
            return None

        input_tensors = []
        for index in range(2):
            input_node = self.get_node(node.args[index])
            input_tensors.append(
                self.define_tensor(
                    input_node,
                    node,
                    self.get_tensor(input_node, node),
                    PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                    nodes_to_wrappers,
                )
            )
        output_tensor = self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        mod_op = PyQnnManager.PyQnnOpWrapper(
            node.name, QNN_OP_PACKAGE_NAME_QTI_AISW, OpElementWiseMod.op_name
        )
        mod_op.AddInputTensors(input_tensors)
        mod_op.AddOutputTensors([output_tensor])
        return mod_op
