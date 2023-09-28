# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, cast
import torch
from executorch.backends.qualcomm.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.qualcomm.utils.qnn_constants import (
    QNN_OP_PACKAGE_NAME_QTI_AISW,
    QNN_OP_POOL_AVG_2D,
    QNN_OP_POOL_AVG_2D_PARAM_FILTER_SIZE,
    QNN_OP_POOL_AVG_2D_PARAM_STRIDE,
    QNN_OP_POOL_AVG_2D_PARAM_PAD_AMOUNT,
    QNN_OP_POOL_AVG_2D_PARAM_COUNT_PAD_FOR_EDGES,
    QNN_OP_POOL_AVG_2D_PARAM_ROUNDING_MODE,
    QNN_OP_POOL_AVG_2D_ROUNDING_MODE_FLOOR,
    QNN_OP_POOL_AVG_2D_ROUNDING_MODE_CEIL,
)
from executorch.backends.qualcomm.utils.utils import get_input_node
import numpy as np

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper


@register_node_visitor
class AvgPool2d(NodeVisitor):
    target = "aten.avg_pool2d.default"

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

        output_tensor, _ = self.get_tensor_shape(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
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

        # if cail mode is True, use ceil instead of floor to compute the output shape
        mode = QNN_OP_POOL_AVG_2D_ROUNDING_MODE_FLOOR
        if len(node.args) > 4:
            ceil_mode = cast(bool, node.args[4])
            if ceil_mode:
                mode = QNN_OP_POOL_AVG_2D_ROUNDING_MODE_CEIL

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
            node.name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_POOL_AVG_2D
        )
        avg_pool2d_op.AddInputTensors([input_tensor_wrapper])
        avg_pool2d_op.AddOutputTensors([output_tensor_wrapper])

        avg_pool2d_op.AddTensorParam(
            QNN_OP_POOL_AVG_2D_PARAM_FILTER_SIZE,
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
            QNN_OP_POOL_AVG_2D_PARAM_STRIDE,
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
            QNN_OP_POOL_AVG_2D_PARAM_PAD_AMOUNT,
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
            QNN_OP_POOL_AVG_2D_PARAM_ROUNDING_MODE,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {"data": np.uint32(mode)},
        )
        avg_pool2d_op.AddScalarParam(
            QNN_OP_POOL_AVG_2D_PARAM_COUNT_PAD_FOR_EDGES,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
            {"data": count_include_pad},
        )

        return avg_pool2d_op
