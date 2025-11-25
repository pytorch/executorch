# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_AXIS_ORDER, QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpReduceMin, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class MinDim(NodeVisitor):
    target = ["aten.min.dim"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> List[PyQnnManager.PyQnnOpWrapper]:
        input_node = self.get_node(node.args[0])
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        # QNN does not support multiple outputs for a single op.
        # Since torch.min(input, dim) returns both values and indices,
        # we only support the value output for OpReduceMin. The index output will be handled
        # separately by OpArgmin.
        # Therefore, we update node.meta["val"] to only keep the value part.
        if len(node.meta["val"]) == 2:
            node.meta["val"] = node.meta["val"][0]

        output_tensor = self.get_tensor(node, node)
        out_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        dims = cast(List[int], [node.args[1]])
        dims = [min_dim % len(input_node.meta["val"].shape) for min_dim in dims]
        if QCOM_AXIS_ORDER in node.meta:
            dims = [node.meta[QCOM_AXIS_ORDER].index(min_dim) for min_dim in dims]
        dims_shape = [len(dims)]

        reduce_min_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpReduceMin.op_name,
        )
        reduce_min_op.AddInputTensors([input_tensor_wrapper])
        reduce_min_op.AddOutputTensors([out_tensor_wrapper])

        reduce_min_op.AddTensorParam(
            OpReduceMin.param_axes,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(dims_shape),
            dims_shape,
            np.array(dims, dtype=np.uint32),
            True,
        )
        if len(node.args) > 2:
            keep_dims = cast(bool, node.args[2])
            reduce_min_op.AddScalarParam(
                OpReduceMin.param_keep_dims,
                PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
                {QCOM_DATA: keep_dims},
            )

        return reduce_min_op
