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
    OpTranspose,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)
from executorch.backends.qualcomm.utils.utils import get_input_node


@register_node_visitor
class TransposeVisitor(NodeVisitor):
    target = "aten.permute_copy.default"

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = get_input_node(node, 0)
        permute_node = input_node if "qnn_permute" in node.meta else node
        input_tensor, use_memo = self.get_tensor_shape(input_node, permute_node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers if use_memo else {},
        )

        # permutation
        permute_order = cast(List[int], node.args[1])
        permute_order_shape = [len(permute_order)]

        output_tensor = input_tensor.permute(permute_order)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        transpose_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpTranspose.op_name,
        )

        # add input/output tensors
        transpose_op.AddInputTensors([input_tensor_wrapper])
        transpose_op.AddOutputTensors([output_tensor_wrapper])

        transpose_op.AddTensorParam(
            OpTranspose.param_perm,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(permute_order_shape),
            permute_order_shape,
            np.array(permute_order, dtype=np.uint32),
            True,
        )
        return transpose_op
