# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpPoolAvg2d, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class AvgPool2d(NodeVisitor):
    target = ["aten.avg_pool2d.default"]

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
        # kernel info
        filter_size = cast(List[int], node.args[1])
        if len(filter_size) == 1:
            filter_size = filter_size + filter_size
        filter_size_shape = [len(filter_size)]

        # stride info
        stride = cast(List[int], node.args[2])
        if len(stride) == 1:
            stride = stride + stride
        stride_shape = [len(stride)]

        padding = [0, 0]
        if len(node.args) > 3:
            padding = cast(List[int], node.args[3])
            if len(padding) == 1:
                padding = padding + padding
        padding_shape = [len(padding), len(padding)]

        # if ceil mode is True, use ceil instead of floor to compute the output shape
        mode = OpPoolAvg2d.RoundingMode.FLOOR
        if len(node.args) > 4:
            ceil_mode = cast(bool, node.args[4])
            if ceil_mode:
                mode = OpPoolAvg2d.RoundingMode.CEIL

        count_include_pad = True
        if len(node.args) > 5:
            count_include_pad = cast(bool, node.args[5])
        # TODO: If count_include_pad = False, it seems not to compute average with padding in Qnn.
        # But it still compute average with padding value, and change divisor in torch
        # if not count_include_pad:
        #     print("Not support count_include_pad = False.")
        #     return

        pooling_region = filter_size[0] * filter_size[1]
        divisor_override = pooling_region  # Default divisor is pooling_region
        if len(node.args) > 6:
            divisor_override = cast(int, node.args[6])
        if divisor_override != pooling_region:
            print("Not support divisor_override which is not equal to pooling region.")
            return

        avg_pool2d_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpPoolAvg2d.op_name,
        )
        avg_pool2d_op.AddInputTensors([input_tensor_wrapper])
        avg_pool2d_op.AddOutputTensors([output_tensor_wrapper])

        avg_pool2d_op.AddTensorParam(
            OpPoolAvg2d.param_filter_size,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(filter_size_shape),
            filter_size_shape,
            np.array(
                filter_size,
                dtype=np.uint32,
            ),
            True,
        )
        avg_pool2d_op.AddTensorParam(
            OpPoolAvg2d.param_stride,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(stride_shape),
            stride_shape,
            np.array(
                stride,
                dtype=np.uint32,
            ),
            True,
        )
        avg_pool2d_op.AddTensorParam(
            OpPoolAvg2d.param_pad_amount,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(padding_shape),
            padding_shape,
            np.array(
                [[padding[0], padding[0]], [padding[1], padding[1]]],
                dtype=np.uint32,
            ),
            True,
        )

        avg_pool2d_op.AddScalarParam(
            OpPoolAvg2d.param_rounding_mode,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(mode)},
        )
        avg_pool2d_op.AddScalarParam(
            OpPoolAvg2d.param_count_pad_for_edges,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
            {QCOM_DATA: count_include_pad},
        )

        return avg_pool2d_op
