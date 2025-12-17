# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import torch

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor


@register_node_visitor
class FullLike(NodeVisitor):
    target = ["aten.full_like.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        in_tensor = node.args[0].meta["val"]
        ref_tensor = torch.zeros(in_tensor.shape, dtype=in_tensor.dtype)
        out_tensor = torch.full_like(ref_tensor, node.args[1])

        # since we can derive the constant value of current op in AoT stage
        # we only build static tensor here for consumers of current node
        # to correctly reference the data
        self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )
