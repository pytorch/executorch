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
from .qnn_constants import OpElementWiseSelect, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Where(NodeVisitor):
    target = ["aten.where.self"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        conditional_input_node = self.get_node(node.args[0])
        conditional_input_tensor = self.get_tensor(conditional_input_node, node)
        conditional_input_tensor_wrapper = self.define_tensor(
            conditional_input_node,
            node,
            conditional_input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        true_input_node = self.get_node(node.args[1])
        true_input_tensor = self.get_tensor(true_input_node, node)
        true_input_tensor_wrapper = self.define_tensor(
            true_input_node,
            node,
            true_input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        false_input_node = self.get_node(node.args[2])
        false_input_tensor = self.get_tensor(false_input_node, node)
        false_input_tensor_wrapper = self.define_tensor(
            false_input_node,
            node,
            false_input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        where_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpElementWiseSelect.op_name,
        )
        where_op.AddInputTensors(
            [
                conditional_input_tensor_wrapper,
                true_input_tensor_wrapper,
                false_input_tensor_wrapper,
            ]
        )
        where_op.AddOutputTensors([output_tensor_wrapper])

        return where_op
