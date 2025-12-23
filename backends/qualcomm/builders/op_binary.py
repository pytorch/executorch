# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA
from executorch.exir.dialects._ops import ops as exir_ops

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpElementWiseBinary, QNN_OP_PACKAGE_NAME_QTI_AISW


# Refer to QnnOpDef.h for the value.
QNN_BINARY_OPERATOR = {
    exir_ops.edge.aten.floor_divide.default: 4,
}


@register_node_visitor
class Binary(NodeVisitor):
    target = ["aten.floor_divide.default"]

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
        binary_output_tensors = [output_tensor_wrapper]

        binary_input_tensors = []
        for index in range(2):
            input_node = self.get_node(node.args[index])
            input_tensor = self.get_tensor(input_node, node)
            tensor_type = PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE

            input_tensor_wrapper = self.define_tensor(
                input_node,
                node,
                input_tensor,
                tensor_type,
                nodes_to_wrappers,
            )
            binary_input_tensors.append(input_tensor_wrapper)

        binary_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpElementWiseBinary.op_name,
        )
        binary_op.AddInputTensors(binary_input_tensors)
        binary_op.AddOutputTensors(binary_output_tensors)

        if node.target not in QNN_BINARY_OPERATOR:
            warnings.warn(
                "[QNN Delegate Op Builder]: This binary operator is not yet supported.",
                stacklevel=1,
            )
            return None

        binary_op.AddScalarParam(
            OpElementWiseBinary.param_operation,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(QNN_BINARY_OPERATOR[node.target])},
        )

        return binary_op
