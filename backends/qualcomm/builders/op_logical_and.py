# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import torch

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpElementWiseAnd, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class And(NodeVisitor):
    target = ["aten.logical_and.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node_1 = self.get_node(node.args[0])
        input_tensor_1 = self.get_tensor(input_node_1, node)
        input_tensor_wrapper_1 = self.define_tensor(
            input_node_1,
            node,
            input_tensor_1,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        input_node_2 = self.get_node(node.args[1])
        input_tensor_2 = self.get_tensor(input_node_2, node)
        input_tensor_wrapper_2 = self.define_tensor(
            input_node_2,
            node,
            input_tensor_2,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        logical_and_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpElementWiseAnd.op_name,
        )
        logical_and_op.AddInputTensors([input_tensor_wrapper_1, input_tensor_wrapper_2])
        logical_and_op.AddOutputTensors([output_tensor_wrapper])

        return logical_and_op
