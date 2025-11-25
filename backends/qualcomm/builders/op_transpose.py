# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_INSERTED_PERMUTE

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpTranspose, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class TransposeVisitor(NodeVisitor):
    target = ["aten.permute_copy.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        input_node = self.get_node(node.args[0])
        permute_node = input_node if QCOM_INSERTED_PERMUTE in node.meta else node
        input_tensor = self.get_tensor(input_node, permute_node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        # permutation
        permute_order = cast(List[int], node.args[1])
        # to prevent negative values
        permute_order = [x % len(permute_order) for x in permute_order]
        permute_order_shape = [len(permute_order)]

        output_tensor = input_tensor.permute(permute_order)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        transpose_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpTranspose.op_name,
        )

        # add input/output tensors
        transpose_op.AddInputTensors([input_tensor_wrapper])
        transpose_op.AddOutputTensors([output_tensor_wrapper])

        transpose_op.AddTensorParam(
            OpTranspose.param_perm,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(permute_order_shape),
            permute_order_shape,
            np.array(permute_order, dtype=np.uint32),
            True,
        )
        return transpose_op
