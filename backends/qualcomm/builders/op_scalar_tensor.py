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
class ScalarTensor(NodeVisitor):
    target = ["scalar_tensor.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        val = node.args[0]
        out_tensor = torch.tensor([val], dtype=node.meta["val"].dtype)

        # The following clamping will only occur in FP mode. Clamping for quantized mode will happen in the pass ReplaceInfValues.
        # negative infinite
        if torch.isinf(out_tensor)[0] and (out_tensor < 0):
            out_tensor = torch.tensor(
                [torch.finfo(torch.float32).min], dtype=node.meta["val"].dtype
            )
        # positive infinite
        elif torch.isinf(out_tensor)[0] and (out_tensor > 0):
            out_tensor = torch.tensor(
                [torch.finfo(torch.float32).max], dtype=node.meta["val"].dtype
            )
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
