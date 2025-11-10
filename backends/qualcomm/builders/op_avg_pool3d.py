# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import numpy as np

import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpPoolAvg3d, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class AvgPool3d(NodeVisitor):
    target = ["aten.avg_pool3d.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

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

        # kernel info
        filter_size = cast(List[int], node.args[1])
        if len(filter_size) == 1:
            filter_size *= 3
        filter_size_shape = [len(filter_size)]

        # stride info
        stride = cast(List[int], node.args[2])
        if len(stride) == 1:
            stride *= 3
        stride_shape = [len(stride)]

        # padding info
        padding = [0, 0, 0]
        if len(node.args) > 3:
            padding = cast(List[int], node.args[3])
            if len(padding) == 1:
                padding *= 3

        # if ceil mode is True, use ceil instead of floor to compute the output shape
        mode = OpPoolAvg3d.RoundingMode.FLOOR
        if len(node.args) > 4:
            ceil_mode = cast(bool, node.args[4])
            if ceil_mode:
                mode = OpPoolAvg3d.RoundingMode.CEIL

        count_pad_for_edges = node.args[5] if len(node.args) > 5 else False

        # pad left, pad right
        depth_pad_l = padding[0]
        depth_pad_r = padding[0]
        height_pad_l = padding[1]
        height_pad_r = padding[1]
        width_pad_l = padding[2]
        width_pad_r = padding[2]

        shape_pad = [
            [depth_pad_l, depth_pad_r],
            [height_pad_l, height_pad_r],
            [width_pad_l, width_pad_r],
        ]
        padding_shape = [len(shape_pad), len(shape_pad[0])]

        out_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        avg_pool3d_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpPoolAvg3d.op_name,
        )

        avg_pool3d_op.AddInputTensors([input_tensor_wrapper])
        avg_pool3d_op.AddOutputTensors([output_tensor_wrapper])

        avg_pool3d_op.AddTensorParam(
            OpPoolAvg3d.param_filter_size,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(filter_size_shape),
            filter_size_shape,
            np.array(
                filter_size,
                dtype=np.uint32,
            ),
            True,
        )

        avg_pool3d_op.AddTensorParam(
            OpPoolAvg3d.param_stride,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(stride_shape),
            stride_shape,
            np.array(
                stride,
                dtype=np.uint32,
            ),
            True,
        )

        avg_pool3d_op.AddTensorParam(
            OpPoolAvg3d.param_pad_amount,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(padding_shape),
            padding_shape,
            np.array(
                shape_pad,
                dtype=np.uint32,
            ),
            True,
        )

        avg_pool3d_op.AddScalarParam(
            OpPoolAvg3d.param_count_pad_for_edges,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
            {QCOM_DATA: count_pad_for_edges},
        )

        avg_pool3d_op.AddScalarParam(
            OpPoolAvg3d.param_rounding_mode,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(mode)},
        )

        return avg_pool3d_op


@register_node_visitor
class AdaptiveAvgPool3d(NodeVisitor):
    target = ["aten._adaptive_avg_pool3d.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

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
        # NOTE: This operator is layout sensitive, so the input tensor shape is always N,D,H,W,C.
        input_depth = input_tensor.shape[1]
        input_height = input_tensor.shape[2]
        input_width = input_tensor.shape[3]
        output_depth = node.args[1][0]
        output_height = node.args[1][1]
        output_width = node.args[1][2]
        if output_depth is None:
            output_depth = input_depth
        if output_height is None:
            output_height = input_height
        if output_width is None:
            output_width = input_width

        # kernel info & stride info
        stride_height = input_height // output_height
        filter_height = input_height - (output_height - 1) * stride_height
        stride_width = input_width // output_width
        filter_width = input_width - (output_width - 1) * stride_width
        stride_depth = input_depth // output_depth
        filter_depth = input_depth - (output_depth - 1) * stride_depth

        filter_size = [filter_depth, filter_height, filter_width]
        filter_shape = [len(filter_size)]
        stride = [stride_depth, stride_height, stride_width]
        stride_shape = [len(stride)]

        depth = (output_depth - 1) * stride_depth + filter_depth - input_depth
        height = (output_height - 1) * stride_height + filter_height - input_height
        width = (output_width - 1) * stride_width + filter_width - input_width

        if any(x != 0 for x in (depth, height, width)):
            warnings.warn(
                "[QNN Delegate Op Builder]: Depth or Height or Width is not suitable, fallback op",
                stacklevel=1,
            )
            return

        count_pad_for_edges = False
        # This operator use the default rounding mode of avg_pool3d, floor.
        mode = OpPoolAvg3d.RoundingMode.FLOOR

        # pad left, pad right, use default 0
        depth_pad_b = 0
        depth_pad_a = 0
        height_pad_b = 0
        height_pad_a = 0
        width_pad_b = 0
        width_pad_a = 0

        shape_pad = [
            [depth_pad_b, depth_pad_a],
            [height_pad_b, height_pad_a],
            [width_pad_b, width_pad_a],
        ]
        padding_shape = [len(shape_pad), len(shape_pad[0])]

        out_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        adaptive_avg_pool3d_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpPoolAvg3d.op_name,
        )

        adaptive_avg_pool3d_op.AddInputTensors([input_tensor_wrapper])
        adaptive_avg_pool3d_op.AddOutputTensors([output_tensor_wrapper])

        adaptive_avg_pool3d_op.AddTensorParam(
            OpPoolAvg3d.param_filter_size,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(filter_shape),
            filter_shape,
            np.array(
                filter_size,
                dtype=np.uint32,
            ),
            True,
        )

        adaptive_avg_pool3d_op.AddTensorParam(
            OpPoolAvg3d.param_stride,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(stride_shape),
            stride_shape,
            np.array(
                stride,
                dtype=np.uint32,
            ),
            True,
        )

        adaptive_avg_pool3d_op.AddTensorParam(
            OpPoolAvg3d.param_pad_amount,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(padding_shape),
            padding_shape,
            np.array(
                shape_pad,
                dtype=np.uint32,
            ),
            True,
        )

        adaptive_avg_pool3d_op.AddScalarParam(
            OpPoolAvg3d.param_count_pad_for_edges,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
            {QCOM_DATA: count_pad_for_edges},
        )

        adaptive_avg_pool3d_op.AddScalarParam(
            OpPoolAvg3d.param_rounding_mode,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(mode)},
        )

        return adaptive_avg_pool3d_op
