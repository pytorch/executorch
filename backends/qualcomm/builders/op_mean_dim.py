# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import numpy as np

import torch
from executorch.backends.qualcomm.builders.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.qualcomm.utils.qnn_constants import (
    OpReduceMean,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)
from executorch.backends.qualcomm.utils.utils import get_input_node


@register_node_visitor
class MeanDim(NodeVisitor):
    target = "aten.mean.dim"

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

        # mean dims and keep dims
        mean_dims = cast(List[int], node.args[1])
        mean_dims = [
            mean_dim % len(input_node.meta["val"].shape) for mean_dim in mean_dims
        ]
        if "axis_order" in node.meta:
            mean_dims = [
                node.meta["axis_order"].index(mean_dim) for mean_dim in mean_dims
            ]
        mean_dims_shape = [len(mean_dims)]

        output_tensor, _ = self.get_tensor_shape(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        reduce_mean_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpReduceMean.op_name,
        )
        reduce_mean_op.AddInputTensors([input_tensor_wrapper])
        reduce_mean_op.AddOutputTensors([output_tensor_wrapper])
        reduce_mean_op.AddTensorParam(
            OpReduceMean.param_axes,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(mean_dims_shape),
            mean_dims_shape,
            np.array(mean_dims, dtype=np.uint32),
            True,
        )
        if len(node.args) > 2:
            keep_dims = cast(bool, node.args[2])
            reduce_mean_op.AddScalarParam(
                OpReduceMean.param_keep_dims,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
                {"data": keep_dims},
            )

        return reduce_mean_op
