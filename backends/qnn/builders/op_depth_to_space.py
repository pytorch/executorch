# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Dict

import torch
from executorch.backends.qnn.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.qnn.utils.qnn_constants import (
    QNN_OP_PACKAGE_NAME_QTI_AISW,
    QNN_OP_DEPTH_TO_SPACE,
    QNN_OP_DEPTH_TO_SPACE_PARAM_BLOCK_SIZE,
    QNN_OP_DEPTH_TO_SPACE_PARAM_MODE,
    QNN_OP_DEPTH_TO_SPACE_MODE_DCR,
    QNN_OP_DEPTH_TO_SPACE_MODE_CRD,
)

import executorch.backends.qnn.python.PyQnnWrapperAdaptor as PyQnnWrapper
from executorch.backends.qnn.utils.utils import get_input_node


@register_node_visitor
class DepthToSpaceVisitor(NodeVisitor):
    target = "aten.pixel_shuffle.default"

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

        block_size = []
        for index in range(1, 3):
            block_size.append(
                output_tensor.shape[index] / input_tensor.shape[index]
            )
        block_size = np.array(block_size, dtype=np.uint32)    
        block_size_shape = [2]

        depth_to_space_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_DEPTH_TO_SPACE
        )
        depth_to_space_op.AddInputTensors([input_tensor_wrapper])
        depth_to_space_op.AddOutputTensors([output_tensor_wrapper])
        depth_to_space_op.AddTensorParam(
            QNN_OP_DEPTH_TO_SPACE_PARAM_BLOCK_SIZE,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(block_size.shape),
            block_size_shape,
            block_size,
            True,
        )
        depth_to_space_op.AddScalarParam(
            QNN_OP_DEPTH_TO_SPACE_PARAM_MODE,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {"data": QNN_OP_DEPTH_TO_SPACE_MODE_CRD},
        )

        return depth_to_space_op
