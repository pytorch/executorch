# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import OpTile, QNN_OP_PACKAGE_NAME_QTI_AISW


@register_node_visitor
class Expand(NodeVisitor):
    target = ["aten.expand_copy.default"]

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

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
            is_input_tensor=False,
        )

        sizes = cast(List[int], node.args[1])

        shape = input_tensor.shape
        input_dims = len(input_tensor.size())
        output_dims = len(output_tensor.size())

        if input_dims < output_dims:
            print(
                f"The rank of input tensor: {input_dims} is less than the rank of output tensor: {output_dims}."
            )
            return

        multiples = [1] * input_dims
        multiples_shape = [input_dims]
        for i in range(input_dims):
            if sizes[i] != -1 and shape[i] == 1:
                multiples[i] = sizes[i]

        tile_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpTile.op_name,
        )
        tile_op.AddInputTensors([input_tensor_wrapper])
        tile_op.AddOutputTensors([output_tensor_wrapper])
        tile_op.AddTensorParam(
            OpTile.param_multiples,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            len(multiples_shape),
            multiples_shape,
            np.array(multiples, dtype=np.uint32),
            True,
        )
        return tile_op
