# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpGatherElements, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Gather(NodeVisitor):
    target = ["aten.gather.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        dim = cast(int, node.args[1])

        indices_node = node.args[2]
        indices_tensor = self.get_tensor(indices_node, node)
        indices_tensor_wrapper = self.define_tensor(
            indices_node,
            node,
            indices_tensor.to(torch.int32),
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        gather_input_tensors = [input_tensor_wrapper, indices_tensor_wrapper]
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
            OpGatherElements.op_name,
        )
        gather_op.AddInputTensors(gather_input_tensors)
        gather_op.AddOutputTensors(gather_output_tensors)
        gather_op.AddScalarParam(
            OpGatherElements.param_axis,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(dim)},
        )

        return gather_op
