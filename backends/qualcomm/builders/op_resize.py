# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import numpy as np
import torch

from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpResize, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Resize(NodeVisitor):
    # Because QNN support ResizeBilinear and ResizeNearestNeighbor, only bicubic need to be handled in resize op
    target = ["aten.upsample_bicubic2d.vec"]

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
        align_corners = cast(bool, node.args[2])
        transformation_mode = np.uint32(2) if align_corners else np.uint32(1)
        # This builder supports only bicubic resize.
        interpolation_mode = np.uint32(2)
        cubic_coeff = np.float32(-0.75)

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        resize_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpResize.op_name,
        )
        resize_op.AddInputTensors([input_tensor_wrapper])
        resize_op.AddOutputTensors([output_tensor_wrapper])

        resize_op.AddScalarParam(
            OpResize.param_exclude_outside,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
            {QCOM_DATA: False},
        )
        resize_op.AddScalarParam(
            OpResize.param_transformation_mode,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: transformation_mode},
        )

        resize_op.AddScalarParam(
            OpResize.param_interpolation_mode,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: interpolation_mode},
        )
        resize_op.AddScalarParam(
            OpResize.param_cubic_coeff,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {QCOM_DATA: cubic_coeff},
        )

        return resize_op
