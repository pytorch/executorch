# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import torch

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpMatMul, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Matmul(NodeVisitor):
    target = "aten.matmul.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        matmul_output_tensors = [output_tensor_wrapper]

        matmul_input_tensors = []
        for index in range(2):
            input_node = node.args[index]
            input_tensor = self.get_tensor(input_node, node)

            # For constant input, the size of tensor is torch.Size([])
            if len(input_tensor.shape) == 0:
                input_tensor = input_tensor.expand(output_tensor.shape).contiguous()

            input_tensor_wrapper = self.define_tensor(
                input_node,
                input_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers,
            )
            matmul_input_tensors.append(input_tensor_wrapper)

        matmul_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name, QNN_OP_PACKAGE_NAME_QTI_AISW, OpMatMul.op_name
        )
        matmul_op.AddInputTensors(matmul_input_tensors)
        matmul_op.AddOutputTensors(matmul_output_tensors)

        return matmul_op
