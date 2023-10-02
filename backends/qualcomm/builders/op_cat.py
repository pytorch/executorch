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
    OpConcat,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)


@register_node_visitor
class Cat(NodeVisitor):
    target = "aten.cat.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        list_of_tensors = cast(List[torch.fx.Node], node.args[0])
        list_of_tensor_wrappers = []

        for tensor_input in list_of_tensors:
            input_tensor, use_memo = self.get_tensor_shape(tensor_input, node)
            list_of_tensor_wrappers.append(
                self.define_tensor(
                    tensor_input,
                    input_tensor,
                    PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                    nodes_to_wrappers if use_memo else {},
                )
            )

        if len(list_of_tensors) != len(list_of_tensor_wrappers):
            print(
                "The number or input tensors is not equal to the number of input tensor wrappers."
            )
            return

        output_tensor, _ = self.get_tensor_shape(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        axis = cast(int, node.args[1])
        if "axis_order" in node.meta:
            axis = node.meta["axis_order"].index(axis)

        concat_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpConcat.op_name,
        )
        concat_op.AddInputTensors(list_of_tensor_wrappers)
        concat_op.AddOutputTensors([output_tensor_wrapper])

        concat_op.AddScalarParam(
            OpConcat.param_axis,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            {"data": np.uint32(axis)},
        )

        return concat_op
