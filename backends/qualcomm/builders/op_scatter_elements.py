# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_AXIS_ORDER, QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpScatterElements, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class ScatterElements(NodeVisitor):
    target = ["aten.scatter.src"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        index_node = self.get_node(node.args[2])
        index_tensor = self.get_tensor(index_node, node)
        index_tensor_wrapper = self.define_tensor(
            index_node,
            node,
            index_tensor.to(torch.int32),
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        updates_node = self.get_node(node.args[3])
        updates_tensor = self.get_tensor(updates_node, node)
        updates_tensor_wrapper = self.define_tensor(
            updates_node,
            node,
            updates_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        dim = node.args[1]
        if dim < 0:
            dim = dim % len(input_tensor.shape)

        if QCOM_AXIS_ORDER in node.meta:
            dim = node.meta[QCOM_AXIS_ORDER].index(dim)

        scatter_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpScatterElements.op_name,
        )
        scatter_op.AddInputTensors(
            [
                input_tensor_wrapper,
                index_tensor_wrapper,
                updates_tensor_wrapper,
            ]
        )
        scatter_op.AddOutputTensors([output_tensor_wrapper])

        scatter_op.AddScalarParam(
            OpScatterElements.param_axis,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(dim)},
        )

        scatter_op.AddScalarParam(
            OpScatterElements.param_reduction,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(OpScatterElements.Reduction.NONE)},
        )

        return scatter_op
