# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import cast, Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpStridedSlice, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class SelectCopy(NodeVisitor):
    target = ["aten.select_copy.int", "aten.select.int"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=True,
        )

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )

        dim = cast(int, node.args[1])
        if dim < 0:
            dim = dim % len(input_tensor.shape)
        index = cast(int, node.args[2]) % input_tensor.shape[dim]

        input_tensor_rank = len(input_tensor.shape)
        ranges = []
        for i in range(input_tensor_rank):
            if i == dim:
                ranges.extend([index, index, 1])
            else:
                ranges.extend([0, input_tensor.shape[i], 1])

        range_shape = [input_tensor_rank, 3]

        stride_slice_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpStridedSlice.op_name,
        )
        stride_slice_op.AddInputTensors([input_tensor_wrapper])
        stride_slice_op.AddOutputTensors([output_tensor_wrapper])

        stride_slice_op.AddTensorParam(
            OpStridedSlice.param_ranges,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            len(range_shape),
            range_shape,
            np.array(ranges, dtype=np.int32),
            True,
        )

        stride_slice_op.AddScalarParam(
            OpStridedSlice.param_shrink_axes,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(math.pow(2, dim))},
        )

        return stride_slice_op
