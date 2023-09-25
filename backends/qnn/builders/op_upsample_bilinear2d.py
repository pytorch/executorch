# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict
import torch
from executorch.backends.qnn.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.qnn.utils.qnn_constants import (
    QNN_OP_PACKAGE_NAME_QTI_AISW,
    QNN_OP_RESIZE_BILINEAR,
    QNN_OP_RESIZE_BILINEAR_ALIGN_CORNERS,
    QNN_OP_RESIZE_BILINEAR_HALF_PIXEL_CENTERS,
)
from executorch.backends.qnn.utils.utils import get_input_node

import executorch.backends.qnn.python.PyQnnWrapperAdaptor as PyQnnWrapper


@register_node_visitor
class ResizeBilinear(NodeVisitor):
    target = "aten.upsample_bilinear2d.default"

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

        reisze_bilinear_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_RESIZE_BILINEAR
        )
        reisze_bilinear_op.AddInputTensors([input_tensor_wrapper])
        reisze_bilinear_op.AddOutputTensors([output_tensor_wrapper])

        reisze_bilinear_op.AddScalarParam(
            QNN_OP_RESIZE_BILINEAR_ALIGN_CORNERS,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
            {"data": node.args[2]},
        )
        reisze_bilinear_op.AddScalarParam(
            QNN_OP_RESIZE_BILINEAR_HALF_PIXEL_CENTERS,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
            {"data": True},
        )

        return reisze_bilinear_op
