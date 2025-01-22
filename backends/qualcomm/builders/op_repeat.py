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
class Repeat(NodeVisitor):
    target = ["aten.repeat.default"]

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
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        multiples = cast(List[int], node.args[1])
        multiples_shape = [len(multiples)]

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
