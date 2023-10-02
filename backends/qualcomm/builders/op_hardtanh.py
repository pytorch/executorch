# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import numpy as np

import torch
from executorch.backends.qualcomm.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.qualcomm.utils.qnn_constants import (
    OpReluMinMax,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)
from executorch.backends.qualcomm.utils.utils import get_input_node


@register_node_visitor
class HardTanhVisitor(NodeVisitor):
    target = "aten.hardtanh.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = get_input_node(node, 0)
        input_tensor, use_memo = self.get_tensor_shape(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers if use_memo else {},
        )

        # default value of output_min and output_max
        output_min = -1
        output_max = 1

        if len(node.args) > 1:
            # update output_min
            output_min = cast(float, node.args[1])
            # update output_max
            output_max = cast(float, node.args[2])

        output_tensor, _ = self.get_tensor_shape(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        hardtanh_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpReluMinMax.op_name,
        )
        hardtanh_op.AddInputTensors([input_tensor_wrapper])
        hardtanh_op.AddOutputTensors([output_tensor_wrapper])
        hardtanh_op.AddScalarParam(
            OpReluMinMax.param_max_value,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {"data": np.float32(output_max)},
        )
        hardtanh_op.AddScalarParam(
            OpReluMinMax.param_min_value,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {"data": np.float32(output_min)},
        )

        return hardtanh_op
