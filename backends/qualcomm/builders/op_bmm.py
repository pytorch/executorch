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
    OpMatMul,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)
from executorch.backends.qualcomm.utils.utils import get_input_node


@register_node_visitor
class Bmm(NodeVisitor):
    target = "aten.bmm.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        output_tensor, _ = self.get_tensor_shape(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        bmm_output_tensors = [output_tensor_wrapper]

        bmm_input_tensors = []
        for index in range(2):
            input_node = get_input_node(node, index)
            input_tensor, use_memo = self.get_tensor_shape(input_node, node)

            # For constant input, the size of tensor is torch.Size([])
            if len(input_tensor.shape) == 0:
                input_tensor = input_tensor.expand(output_tensor.shape).contiguous()

            input_tensor_wrapper = self.define_tensor(
                input_node,
                input_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers if use_memo else {},
            )
            bmm_input_tensors.append(input_tensor_wrapper)

        bmm_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpMatMul.op_name,
        )
        bmm_op.AddInputTensors(bmm_input_tensors)
        bmm_op.AddOutputTensors(bmm_output_tensors)

        return bmm_op
