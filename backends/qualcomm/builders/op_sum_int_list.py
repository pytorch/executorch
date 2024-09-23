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

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpReduceSum, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Sum(NodeVisitor):
    target = ["aten.sum.dim_IntList"]

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
        sum_input_tensors = [input_tensor_wrapper]

        # sum dims
        sum_dims = cast(List[int], node.args[1])
        sum_dims = [sum_dim % len(input_node.meta["val"].shape) for sum_dim in sum_dims]
        if QCOM_AXIS_ORDER in node.meta:
            sum_dims = [
                node.meta[QCOM_AXIS_ORDER].index(sum_dim) for sum_dim in sum_dims
            ]
        sum_dims_shape = [len(sum_dims)]

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )
        sum_output_tensors = [output_tensor_wrapper]
        sum_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpReduceSum.op_name,
        )
        sum_op.AddInputTensors(sum_input_tensors)
        sum_op.AddOutputTensors(sum_output_tensors)
        sum_op.AddTensorParam(
            OpReduceSum.param_axes,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(sum_dims_shape),
            sum_dims_shape,
            np.array(sum_dims, dtype=np.uint32),
            True,
        )

        if len(node.args) > 2:
            keep_dims = cast(bool, node.args[2])
            sum_op.AddScalarParam(
                OpReduceSum.param_keep_dims,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
                {QCOM_DATA: keep_dims},
            )
        return sum_op
