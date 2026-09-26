# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpRandomNormalLike, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Randn(NodeVisitor):
    target = ["aten.randn.default", "aten.randn_like.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        output_tensor = node.meta["val"]
        output_shape = list(output_tensor.shape)

        shape_data = np.array(output_shape, dtype=np.uint32)
        shape_dims = [len(output_shape)]

        shape_tensor_wrapper = PyQnnManager.TensorWrapper(
            f"{node.name}_shape",
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            PyQnnManager.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
            {},
            len(shape_dims),
            shape_dims,
            [],
            shape_data,
            True,
        )

        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        randn_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpRandomNormalLike.op_name,
        )

        randn_op.AddInputTensors([shape_tensor_wrapper])
        randn_op.AddOutputTensors([output_tensor_wrapper])

        randn_op.AddScalarParam(
            OpRandomNormalLike.param_mean,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {QCOM_DATA: np.float32(0.0)},
        )

        randn_op.AddScalarParam(
            OpRandomNormalLike.param_scale,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {QCOM_DATA: np.float32(1.0)},
        )

        return randn_op
