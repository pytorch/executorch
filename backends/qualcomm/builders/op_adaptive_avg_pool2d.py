# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import numpy as np

import torch

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpPoolAvg2d, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class AdaptiveAvgPool2D(NodeVisitor):
    target = ["aten.adaptive_avg_pool2d.default"]

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

        input_height = input_tensor.shape[1]
        input_width = input_tensor.shape[2]

        output_height = node.args[1][0]
        output_width = node.args[1][1]

        filter_height = input_height // output_height
        filter_width = input_width // output_width
        filter = [filter_height, filter_width]
        filter_shape = [len(filter)]

        stride_height = filter_height
        stride_width = filter_width
        stride = [stride_height, stride_width]
        stride_shape = [len(stride)]

        height = (output_height - 1) * stride_height + filter_height - input_height
        width = (output_width - 1) * stride_width + filter_width - input_width
        if height % 2 != 0 or width % 2 != 0:
            warnings.warn(
                "[QNN Delegate Op Builder]: Height or Width is not divisble by 2 with no remainder, fall back op",
                stacklevel=1,
            )
            return

        padding_height = height / 2
        padding_width = width / 2
        padding = [padding_height, padding_width]
        padding_shape = [2, 2]

        out_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        adaptive_avg_pool2d_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpPoolAvg2d.op_name,
        )

        adaptive_avg_pool2d_op.AddInputTensors([input_tensor_wrapper])
        adaptive_avg_pool2d_op.AddOutputTensors([output_tensor_wrapper])

        adaptive_avg_pool2d_op.AddTensorParam(
            OpPoolAvg2d.param_filter_size,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(filter_shape),
            filter_shape,
            np.array(
                filter,
                dtype=np.uint32,
            ),
            True,
        )

        adaptive_avg_pool2d_op.AddTensorParam(
            OpPoolAvg2d.param_stride,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(stride_shape),
            stride_shape,
            np.array(
                stride,
                dtype=np.uint32,
            ),
            True,
        )

        adaptive_avg_pool2d_op.AddTensorParam(
            OpPoolAvg2d.param_pad_amount,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(padding_shape),
            padding_shape,
            np.array(
                [[padding[0], padding[0]], [padding[1], padding[1]]],
                dtype=np.uint32,
            ),
            True,
        )

        return adaptive_avg_pool2d_op
