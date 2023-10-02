# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import torch
from executorch.backends.qualcomm.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.qualcomm.utils.qnn_constants import (
    OpFullyConnected,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)
from executorch.backends.qualcomm.utils.utils import get_input_node


@register_node_visitor
class LinearVisitor(NodeVisitor):
    target = "aten.linear.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        linear_input_tensors = []
        input_node = get_input_node(node, 0)
        input_tensor, _ = self.get_tensor_shape(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        linear_input_tensors.append(input_tensor_wrapper)

        weight_node = get_input_node(node, 1)
        weight_tensor = getattr(weight_node.graph.owning_module, weight_node.target)
        weight_tensor_wrapper = self.define_tensor(
            weight_node,
            weight_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )
        linear_input_tensors.append(weight_tensor_wrapper)

        if len(node.args) >= 3:
            bias_node = get_input_node(node, 2)
            bias_tensor = getattr(bias_node.graph.owning_module, bias_node.target)
            bias_tensor_wrapper = self.define_tensor(
                bias_node,
                bias_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                nodes_to_wrappers,
            )
            linear_input_tensors.append(bias_tensor_wrapper)

        output_tensor, _ = self.get_tensor_shape(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        linear_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpFullyConnected.op_name,
        )
        linear_op.AddInputTensors(linear_input_tensors)
        linear_op.AddOutputTensors([output_tensor_wrapper])

        return linear_op
