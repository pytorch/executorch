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
from .qnn_constants import OpGather, QNN_OP_PACKAGE_NAME_QTI_AISW
from .utils import get_parameter


@register_node_visitor
class Embedding(NodeVisitor):
    target = ["aten.embedding.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        weight_node = self.get_node(node.args[0])
        weight_tensor = get_parameter(weight_node, self.edge_program)
        weight_tensor_wrapper = self.define_tensor(
            weight_node,
            node,
            weight_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        indices_node = node.args[1]
        indices_tensor = self.get_tensor(indices_node, node)
        indices_tensor_wrapper = self.define_tensor(
            indices_node,
            node,
            indices_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        gather_input_tensors = [weight_tensor_wrapper, indices_tensor_wrapper]

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        gather_output_tensors = [output_tensor_wrapper]

        gather_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpGather.op_name,
        )
        gather_op.AddInputTensors(gather_input_tensors)
        gather_op.AddOutputTensors(gather_output_tensors)

        # For now, default axis is zero.
        gather_op.AddScalarParam(
            OpGather.param_axis,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(0)},
        )

        return gather_op
