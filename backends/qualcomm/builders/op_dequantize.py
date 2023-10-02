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
    OpDequantize,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)
from executorch.backends.qualcomm.utils.utils import get_input_node


class DequantizeOpBase(NodeVisitor):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        dequant_input_tensors = []
        input_node = get_input_node(node, 0)
        input_tensor, _ = self.get_tensor_shape(input_node, node)
        inp_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        dequant_input_tensors.append(inp_tensor_wrapper)

        output_tensor, _ = self.get_tensor_shape(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        dequant_output_tensors = [output_tensor_wrapper]

        dequant_op = PyQnnWrapper.PyQnnOpWrapper(
            node.target.__name__,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpDequantize.op_name,
        )
        dequant_op.AddInputTensors(dequant_input_tensors)
        dequant_op.AddOutputTensors(dequant_output_tensors)

        return dequant_op


@register_node_visitor
class PerTensorDequantizeDefault(DequantizeOpBase):
    target = "quantized_decomposed.dequantize_per_tensor.default"


@register_node_visitor
class PerTensorDequantizeTensor(DequantizeOpBase):
    target = "quantized_decomposed.dequantize_per_tensor.tensor"


@register_node_visitor
class PerChannelDequantizeDefault(DequantizeOpBase):
    target = "quantized_decomposed.dequantize_per_channel.default"


@register_node_visitor
class PerChannelDequantizeTensor(DequantizeOpBase):
    target = "quantized_decomposed.dequantize_per_channel.tensor"
