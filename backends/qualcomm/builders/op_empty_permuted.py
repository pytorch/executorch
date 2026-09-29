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
class EmptyPermuted(NodeVisitor):
    target = ["aten.empty_permuted.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        tensor_shape = list(self.get_tensor(node, node).shape)
        out_tensor = torch.zeros(tensor_shape, dtype=node.meta["val"].dtype)

        # the op only reserves uninitialized storage, and the physical layout
        # argument describes strides QNN does not model, so the result can be
        # materialized at AoT stage as a static tensor for consumers of current
        # node to correctly reference the data
        self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )
