# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np

import torch
from executorch.backends.qualcomm.utils.constants import QCOM_AXIS_ORDER, QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpReduceMax, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class AMax(NodeVisitor):
    target = ["aten.amax.default"]

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

        # mean dims and keep dims
        if len(node.args) > 1:
            mean_dims = cast(List[int], node.args[1])
            mean_dims = [
                mean_dim % len(input_node.meta["val"].shape) for mean_dim in mean_dims
            ]
            if QCOM_AXIS_ORDER in node.meta:
                mean_dims = [
                    node.meta[QCOM_AXIS_ORDER].index(mean_dim) for mean_dim in mean_dims
                ]
        else:
            # reduce all dimensions
            mean_dims = list(range(input_node.meta["val"].dim()))

        mean_dims_shape = [len(mean_dims)]

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        reduce_max_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpReduceMax.op_name,
        )
        reduce_max_op.AddInputTensors([input_tensor_wrapper])
        reduce_max_op.AddOutputTensors([output_tensor_wrapper])
        reduce_max_op.AddTensorParam(
            OpReduceMax.param_axes,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(mean_dims_shape),
            mean_dims_shape,
            np.array(mean_dims, dtype=np.uint32),
            True,
        )
        if len(node.args) > 2:
            keep_dims = cast(bool, node.args[2])
            reduce_max_op.AddScalarParam(
                OpReduceMax.param_keep_dims,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
                {QCOM_DATA: keep_dims},
            )

        return reduce_max_op
