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
from .qnn_constants import OpElementWiseNotEqual, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class NotEqual(NodeVisitor):
    target = ["aten.ne.Tensor"]

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
            node,
            out_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        output_tensors = [output_tensor_wrapper]

        input_tensors = []
        for index in range(2):
            input_node = self.get_node(node.args[index])
            input_tensor = self.get_tensor(input_node, node)
            tensor_type = PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE

            input_tensor_wrapper = self.define_tensor(
                input_node,
                node,
                input_tensor,
                tensor_type,
                nodes_to_wrappers,
            )
            input_tensors.append(input_tensor_wrapper)

        ne_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpElementWiseNotEqual.op_name,
        )
        ne_op.AddInputTensors(input_tensors)
        ne_op.AddOutputTensors(output_tensors)

        return ne_op
