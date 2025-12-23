# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpElu, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Elu(NodeVisitor):
    target = ["aten.elu.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        # tensor input
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)

        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        elu_input_tensors = [input_tensor_wrapper]

        out_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        elu_output_tensors = [output_tensor_wrapper]

        elu_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpElu.op_name,
        )
        elu_op.AddInputTensors(elu_input_tensors)
        elu_op.AddOutputTensors(elu_output_tensors)
        if len(node.args) > 1:
            elu_op.AddScalarParam(
                OpElu.param_alpha,
                PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
                {QCOM_DATA: np.float32(node.args[1])},
            )

        return elu_op
