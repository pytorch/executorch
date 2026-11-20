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
from .qnn_constants import OpElementWisePower, QNN_OP_PACKAGE_NAME_QTI_AISW
from .utils import is_parameter


# pow.Tensor_Scalar should fall in this visitor because LiftConstantScalarOperands pass
@register_node_visitor
class PowTensorTensor(NodeVisitor):
    target = ["aten.pow.Tensor_Tensor"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        out_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        pow_output_tensors = [output_tensor_wrapper]

        # tensor input
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        # pow.Scalar's base is lifted to a one-element constant, and quantized
        # HTP ElementWisePower does not broadcast its first input: only the
        # first output element is computed. Materialize the base at full shape.
        if is_parameter(input_node, self.edge_program) and (
            input_tensor.shape != out_tensor.shape
        ):
            input_tensor = input_tensor.expand(out_tensor.shape).contiguous()

        tensor_type = PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE

        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            tensor_type,
            nodes_to_wrappers,
        )

        # exp input
        exp_node = self.get_node(node.args[1])
        exp_tensor = self.get_tensor(exp_node, node)
        exp_tensor_wrapper = self.define_tensor(
            exp_node,
            node,
            exp_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        pow_input_tensors = [input_tensor_wrapper, exp_tensor_wrapper]

        pow_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpElementWisePower.op_name,
        )
        pow_op.AddInputTensors(pow_input_tensors)
        pow_op.AddOutputTensors(pow_output_tensors)

        return pow_op
