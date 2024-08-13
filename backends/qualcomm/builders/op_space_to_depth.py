# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpSpaceToDepth, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class SpaceToDepthVisitor(NodeVisitor):
    target = ["aten.pixel_unshuffle.default"]

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

        block_size = []
        for index in range(1, 3):
            block_size.append(input_tensor.shape[index] / output_tensor.shape[index])
        block_size = np.array(block_size, dtype=np.uint32)
        block_size_shape = [2]

        space_to_depth_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpSpaceToDepth.op_name,
        )
        space_to_depth_op.AddInputTensors([input_tensor_wrapper])
        space_to_depth_op.AddOutputTensors([output_tensor_wrapper])
        space_to_depth_op.AddTensorParam(
            OpSpaceToDepth.param_block_size,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(block_size.shape),
            block_size_shape,
            block_size,
            True,
        )
        space_to_depth_op.AddScalarParam(
            OpSpaceToDepth.param_mode,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(OpSpaceToDepth.Mode.CRD)},
        )

        return space_to_depth_op
