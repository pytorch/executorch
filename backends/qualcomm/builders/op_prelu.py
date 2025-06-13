# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import torch
from executorch.backends.qualcomm.utils.constants import QCOM_AXIS_ORDER

from .node_visitor import get_parameter, NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpPRelu, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class PReLU(NodeVisitor):
    target = ["aten.prelu.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        prelu_inp_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        coeff_node = self.get_node(node.args[1])
        coeff = get_parameter(coeff_node, self.edge_program)
        coeff_tensor = torch.zeros(input_node.meta["val"].shape, dtype=coeff.dtype)
        # per-channel activation
        coeff_node_shape = coeff_node.meta["val"].shape
        if len(coeff_node_shape) and coeff_node_shape[0] > 1:
            for i in range(input_node.meta["val"].shape[1]):
                coeff_tensor = coeff_tensor.index_fill(1, torch.tensor([i]), coeff[i])
        else:
            coeff_tensor.fill_(coeff[0] if coeff.dim() else coeff)

        if axis_order := input_node.meta.get(QCOM_AXIS_ORDER, None):
            coeff_tensor = coeff_tensor.permute(dims=axis_order).contiguous()

        coeff_tensor_wrapper = self.define_tensor(
            coeff_node,
            node,
            coeff_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )
        prelu_input_tensors = [prelu_inp_tensor_wrapper, coeff_tensor_wrapper]

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        prelu_output_tensors = [output_tensor_wrapper]

        prelu_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpPRelu.op_name,
        )
        prelu_op.AddInputTensors(prelu_input_tensors)
        prelu_op.AddOutputTensors(prelu_output_tensors)

        return prelu_op
