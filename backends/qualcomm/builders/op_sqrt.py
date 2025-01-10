# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import torch

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpSqrt, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class SQRT(NodeVisitor):
    target = ["aten.sqrt.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        # tensor input
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)

        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        sqrt_input_tensors = [input_tensor_wrapper]

        out_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        sqrt_output_tensors = [output_tensor_wrapper]

        sqrt_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpSqrt.op_name,
        )
        sqrt_op.AddInputTensors(sqrt_input_tensors)
        sqrt_op.AddOutputTensors(sqrt_output_tensors)

        return sqrt_op
