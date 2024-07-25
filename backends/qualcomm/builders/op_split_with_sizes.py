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
from .qnn_constants import OpSplit, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class SplitWithSizes(NodeVisitor):
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
        input_tensor_wrappers = [input_tensor_wrapper]

        # split_with_sizes will return a tuple since it has multiple outputs
        output_tensor_wrappers = []
        for index in range(len(node.meta["val"])):
            output_tensor = self.get_tensor(node, node, index)
            output_tensor_wrapper = self.define_tensor(
                node,
                output_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers,
                is_input_tensor=False,
                wrapper_idx=index,
            )
            output_tensor_wrappers.append(output_tensor_wrapper)

        chunks = cast(List[int], node.args[1])
        split_indices = []
        sum = 0
        # Edge represents chunks by specifying the size of each chunk
        # QNN represents chunks by specifying the index to split chunks
        for index, _value in enumerate(chunks[:-1]):
            sum = sum + chunks[index]
            split_indices.append(sum)

        split_indices_shape = [len(split_indices)]
        dim = cast(int, node.args[2])
        if dim < 0:
            dim = dim % len(input_tensor.shape)

        if QCOM_AXIS_ORDER in node.meta:
            dim = node.meta[QCOM_AXIS_ORDER].index(dim)
        split_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpSplit.op_name,
        )
        split_op.AddInputTensors(input_tensor_wrappers)
        split_op.AddOutputTensors(output_tensor_wrappers)
        split_op.AddTensorParam(
            OpSplit.param_split_index,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(split_indices_shape),
            split_indices_shape,
            np.array(split_indices, dtype=np.uint32),
            True,
        )

        split_op.AddScalarParam(
            OpSplit.param_axis,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {QCOM_DATA: np.uint32(dim)},
        )
        return split_op
