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
from .qnn_constants import OpCumulativeSum, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class CumulativeSum(NodeVisitor):
    target = ["aten.cumsum.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def get_param(self, node, input_tensor):
        dim = node.args[1]

        if dim < 0:
            dim = dim % len(input_tensor.shape)
        if QCOM_AXIS_ORDER in node.meta:
            dim = node.meta[QCOM_AXIS_ORDER].index(dim)

        return cast(np.uint32, dim)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        dim = self.get_param(node, input_tensor)

        output_tensor = self.get_tensor(node, node)
        if output_tensor.dtype == torch.int64:
            output_tensor = output_tensor.to(torch.int32)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        cumsum_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpCumulativeSum.op_name,
        )
        cumsum_op.AddInputTensors([input_tensor_wrapper])
        cumsum_op.AddOutputTensors([output_tensor_wrapper])
        cumsum_op.AddScalarParam(
            OpCumulativeSum.param_axis,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: dim},
        )
        cumsum_op.AddScalarParam(
            OpCumulativeSum.param_exclusive,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
            {QCOM_DATA: False},
        )
        cumsum_op.AddScalarParam(
            OpCumulativeSum.param_reverse,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
            {QCOM_DATA: False},
        )

        return cumsum_op
