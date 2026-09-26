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
from .qnn_constants import OpChannelShuffle, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class ChannelShuffleVisitor(NodeVisitor):
    target = ["aten.channel_shuffle.default"]

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

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        num_groups = cast(int, node.args[1])
        # QNN ChannelShuffle operates on the channel dimension (axis=1 for NCHW)
        axis = 1

        channel_shuffle_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpChannelShuffle.op_name,
        )
        channel_shuffle_op.AddInputTensors([input_tensor_wrapper])
        channel_shuffle_op.AddOutputTensors([output_tensor_wrapper])
        channel_shuffle_op.AddScalarParam(
            OpChannelShuffle.param_num_groups,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(num_groups)},
        )
        channel_shuffle_op.AddScalarParam(
            OpChannelShuffle.param_axis,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(axis)},
        )

        return channel_shuffle_op
