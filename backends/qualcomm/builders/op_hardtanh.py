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
from .qnn_constants import OpReluMinMax, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class HardTanhVisitor(NodeVisitor):
    target = ["aten.hardtanh.default"]

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

        # default value of output_min and output_max
        output_min = -1
        output_max = 1

        if len(node.args) > 1:
            # update output_min
            output_min = cast(float, node.args[1])
            # update output_max
            output_max = cast(float, node.args[2])

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        hardtanh_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpReluMinMax.op_name,
        )
        hardtanh_op.AddInputTensors([input_tensor_wrapper])
        hardtanh_op.AddOutputTensors([output_tensor_wrapper])
        hardtanh_op.AddScalarParam(
            OpReluMinMax.param_max_value,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {QCOM_DATA: np.float32(output_max)},
        )
        hardtanh_op.AddScalarParam(
            OpReluMinMax.param_min_value,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {QCOM_DATA: np.float32(output_min)},
        )

        return hardtanh_op
