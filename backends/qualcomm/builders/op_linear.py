# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import torch
from executorch.backends.qualcomm.utils.constants import (
    QCOM_QUANT_ATTRS,
    QCOM_SCALES,
    QCOM_ZERO_POINTS,
)

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpFullyConnected, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class LinearVisitor(NodeVisitor):
    target = ["aten.linear.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        linear_input_tensors = []
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        linear_input_tensors.append(input_tensor_wrapper)

        weight_node = self.get_node(node.args[1])
        if (
            quant_attrs := weight_node.meta.get(QCOM_QUANT_ATTRS)
        ) and QCOM_SCALES in quant_attrs:
            # Dimension of weight is [m, n], per channel quant params is [m]
            # Change to [m, 1] to fit the tensor.div(s).add(z)
            quant_attrs[QCOM_SCALES] = quant_attrs[QCOM_SCALES].reshape([-1, 1])
            quant_attrs[QCOM_ZERO_POINTS] = quant_attrs[QCOM_ZERO_POINTS].reshape(
                [-1, 1]
            )
        weight_tensor = self.get_tensor(weight_node, node)
        weight_tensor_wrapper = self.define_tensor(
            weight_node,
            node,
            weight_tensor,
            # It will determine correct QNN tensor type in define_tensor.
            # This param seems unnecessary, which we could possibly remove this in the future.
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        linear_input_tensors.append(weight_tensor_wrapper)

        if len(node.args) >= 3:
            bias_node = self.get_node(node.args[2])
            bias_tensor = self.get_tensor(bias_node, node)
            bias_tensor_wrapper = self.define_tensor(
                bias_node,
                node,
                bias_tensor,
                PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers,
            )
            linear_input_tensors.append(bias_tensor_wrapper)

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        linear_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpFullyConnected.op_name,
        )
        linear_op.AddInputTensors(linear_input_tensors)
        linear_op.AddOutputTensors([output_tensor_wrapper])

        return linear_op
