# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import numpy as np

import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpPoolMax2d, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class AdaptiveMaxPool2D(NodeVisitor):
    target = ["aten.adaptive_max_pool2d.default"]

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
        users = list(node.users.keys())
        for user in users:
            if user.target.__name__ == "getitem":
                getitem_index = user.args[1]
                if getitem_index != 0:
                    warnings.warn(
                        f"[QNN Delegate Op Builder]: Expected second argument of getitem node for {node.target.__name__ } to be 0, got {getitem_index}",
                        stacklevel=1,
                    )
                    return

        if len(node.args) > 2:
            warnings.warn(
                "[QNN Delegate Op Builder]: The return_indices is not supported, fallback op",
                stacklevel=1,
            )
            return

        input_height = input_tensor.shape[1]
        input_width = input_tensor.shape[2]
        # output cases
        out_wh = cast(List[int], node.args[1])
        if len(out_wh) == 1:
            output_height = node.args[1][0]
            output_width = node.args[1][0]
        else:
            output_height = node.args[1][0]
            output_width = node.args[1][1]
        if output_height is None:
            output_height = input_height
        if output_width is None:
            output_width = input_width
        # NOTE: Here we need not to emphasize on mode, cuz the output shape is decided by user.
        mode = OpPoolMax2d.RoundingMode.FLOOR

        # floor division
        stride_height = input_height // output_height
        filter_height = input_height - (output_height - 1) * stride_height
        stride_width = input_width // output_width
        filter_width = input_width - (output_width - 1) * stride_width

        filter = [filter_height, filter_width]
        filter_shape = [len(filter)]

        stride = [stride_height, stride_width]
        stride_shape = [len(stride)]

        padding = [0, 0]
        padding_shape = [len(padding), len(padding)]

        out_tensor = self.get_tensor(node, node, 0)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        adaptive_max_pool2d_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpPoolMax2d.op_name,
        )

        adaptive_max_pool2d_op.AddInputTensors([input_tensor_wrapper])
        adaptive_max_pool2d_op.AddOutputTensors([output_tensor_wrapper])

        adaptive_max_pool2d_op.AddTensorParam(
            OpPoolMax2d.param_filter_size,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(filter_shape),
            filter_shape,
            np.array(
                filter,
                dtype=np.uint32,
            ),
            True,
        )

        adaptive_max_pool2d_op.AddTensorParam(
            OpPoolMax2d.param_stride,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(stride_shape),
            stride_shape,
            np.array(
                stride,
                dtype=np.uint32,
            ),
            True,
        )

        adaptive_max_pool2d_op.AddTensorParam(
            OpPoolMax2d.param_pad_amount,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(padding_shape),
            padding_shape,
            np.array(
                [[padding[0], padding[0]], [padding[1], padding[1]]],
                dtype=np.uint32,
            ),
            True,
        )

        adaptive_max_pool2d_op.AddScalarParam(
            OpPoolMax2d.param_rounding_mode,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(mode)},
        )

        return adaptive_max_pool2d_op
