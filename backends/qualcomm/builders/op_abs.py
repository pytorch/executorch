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
from .qnn_constants import OpElementWiseAbs, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Abs(NodeVisitor):
    target = ["aten.abs.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        out_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        abs_output_tensors = [output_tensor_wrapper]

        input_node = self.get_node(node.args[0])
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            self.get_tensor(input_node, node),
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        abs_input_tensors = [input_tensor_wrapper]

        abs_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpElementWiseAbs.op_name,
        )
        abs_op.AddInputTensors(abs_input_tensors)
        abs_op.AddOutputTensors(abs_output_tensors)

        return abs_op
