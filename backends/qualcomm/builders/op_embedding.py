# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor, register_node_visitor
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
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        weight_node = node.args[0]
        weight_tensor = get_parameter(weight_node, self.edge_program)
        weight_tensor_wrapper = self.define_tensor(
            weight_node,
            weight_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
            is_input_tensor=True,
        )

        indices_node = node.args[1]
        indices_tensor = self.get_tensor(indices_node, node)
        indices_tensor_wrapper = self.define_tensor(
            indices_node,
            indices_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )

        gather_input_tensors = [weight_tensor_wrapper, indices_tensor_wrapper]

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )
        gather_output_tensors = [output_tensor_wrapper]

        gather_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpGather.op_name,
        )
        gather_op.AddInputTensors(gather_input_tensors)
        gather_op.AddOutputTensors(gather_output_tensors)

        # For now, default axis is zero.
        gather_op.AddScalarParam(
            OpGather.param_axis,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(0)},
        )

        return gather_op
