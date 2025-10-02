# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_AXIS_ORDER, QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpPack, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Stack(NodeVisitor):
    target = ["aten.stack.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node_list = node.args[0]
        stack_input_tensors = []
        for input_node in input_node_list:
            input_tensor = self.get_tensor(self.get_node(input_node), node)
            stack_inp_tensor_wrapper = self.define_tensor(
                input_node,
                node,
                input_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers,
            )
            stack_input_tensors.append(stack_inp_tensor_wrapper)
        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        stack_output_tensors = [output_tensor_wrapper]

        dim = 0 if len(node.args) == 1 else cast(int, node.args[1])
        if dim < 0:
            dim = dim % len(output_tensor.shape)
        if QCOM_AXIS_ORDER in node.meta:
            dim = node.meta[QCOM_AXIS_ORDER].index(dim)
        stack_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpPack.op_name,
        )
        stack_op.AddInputTensors(stack_input_tensors)
        stack_op.AddOutputTensors(stack_output_tensors)

        stack_op.AddScalarParam(
            OpPack.param_axis,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(dim)},
        )

        return stack_op
