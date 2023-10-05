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
    OpQuantize,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)
from executorch.backends.qualcomm.utils.utils import get_input_node


class QuantizeOpBase(NodeVisitor):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        quant_input_tensors = []
        input_node = get_input_node(node, 0)
        input_tensor, _ = self.get_tensor_shape(input_node, node)
        inp_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        quant_input_tensors.append(inp_tensor_wrapper)

        node.meta["quant_attrs"] = {"encoding": node.target}
        arg_schemas = list(node.target._schema.arguments)[1:]
        for i, arg_schema in enumerate(arg_schemas):
            name = arg_schema.name
            node.meta["quant_attrs"][name] = get_input_node(node, i + 1)

        output_tensor, _ = self.get_tensor_shape(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        quant_output_tensors = [output_tensor_wrapper]

        quant_op = PyQnnWrapper.PyQnnOpWrapper(
            node.target.__name__,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpQuantize.op_name,
        )
        quant_op.AddInputTensors(quant_input_tensors)
        quant_op.AddOutputTensors(quant_output_tensors)

        return quant_op


@register_node_visitor
class PerTensorQuantize(QuantizeOpBase):
    target = "quantized_decomposed.quantize_per_tensor.default"


@register_node_visitor
class PerChannelQuantize(QuantizeOpBase):
    target = "quantized_decomposed.quantize_per_channel.default"
