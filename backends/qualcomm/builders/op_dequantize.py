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
from .qnn_constants import OpDequantize, QNN_OP_PACKAGE_NAME_QTI_AISW


class DequantizeOpBase(NodeVisitor):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        dequant_input_tensors = []
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)
        inp_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        dequant_input_tensors.append(inp_tensor_wrapper)

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        dequant_output_tensors = [output_tensor_wrapper]

        dequant_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpDequantize.op_name,
        )
        dequant_op.AddInputTensors(dequant_input_tensors)
        dequant_op.AddOutputTensors(dequant_output_tensors)

        return dequant_op


@register_node_visitor
class PerTensorDequantize(DequantizeOpBase):
    target = [
        "quantized_decomposed.dequantize_per_tensor.default",
        "quantized_decomposed.dequantize_per_tensor.tensor",
    ]


@register_node_visitor
class PerChannelDequantize(DequantizeOpBase):
    target = [
        "quantized_decomposed.dequantize_per_channel.default",
        "quantized_decomposed.dequantize_per_channel.tensor",
    ]
