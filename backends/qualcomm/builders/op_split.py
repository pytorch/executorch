# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Dict

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpSplit, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Split(NodeVisitor):
    target = ["aten.split_with_sizes.default"]

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
        split_input_tensors = [input_tensor_wrapper]

        axis = 0 if len(node.args) < 3 else cast(int, node.args[2])
        if axis < 0:
            axis = axis % len(input_tensor.shape)
        if "axis_order" in node.meta:
            axis = node.meta["axis_order"].index(axis)

        # this is not the general case, only a quick workaround here
        index = np.arange(1, input_tensor.shape[axis], dtype=np.uint32)
        index_shape = [len(index)]

        split_output_tensors = []
        for i in range(input_tensor.shape[axis]):
            output_tensor = self.get_tensor(node, node, i)
            output_tensor_wrapper = self.define_tensor(
                node,
                output_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers,
                is_input_tensor=False,
                wrapper_idx=i,
            )
            split_output_tensors.append(output_tensor_wrapper)

        split_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpSplit.op_name,
        )
        split_op.AddInputTensors(split_input_tensors)
        split_op.AddOutputTensors(split_output_tensors)

        split_op.AddScalarParam(
            OpSplit.param_axis,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {"data": np.uint32(axis)},
        )
        split_op.AddTensorParam(
            OpSplit.param_split_index,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(index_shape),
            index_shape,
            index,
            True,
        )

        return split_op
