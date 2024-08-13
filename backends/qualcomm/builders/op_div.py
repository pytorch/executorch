# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import torch

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpElementWiseDivide, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Div(NodeVisitor):
    target = ["aten.div.Tensor"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        out_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            out_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )
        div_output_tensors = [output_tensor_wrapper]

        div_input_tensors = []
        for index in range(2):
            input_node = node.args[index]
            input_tensor = self.get_tensor(input_node, node)
            tensor_type = PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE

            input_tensor_wrapper = self.define_tensor(
                input_node,
                input_tensor,
                tensor_type,
                nodes_to_wrappers,
                is_input_tensor=True,
            )
            div_input_tensors.append(input_tensor_wrapper)

        div_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpElementWiseDivide.op_name,
        )
        div_op.AddInputTensors(div_input_tensors)
        div_op.AddOutputTensors(div_output_tensors)

        return div_op
