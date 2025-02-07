# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional, Union

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class ViewCopyToSqueezeUnsqueezePass(ExportPass):
    """
    Replaces view_copy nodes with squeeze_copy.dims nodes if the view node reduces dims of size 1.
    Replaces view_copy nodes with unsqueeze_copy.default nodes if the view node adds a dim of size 1.
    """

    def __init__(self) -> None:
        super().__init__()
        self.view_copy_op: torch._ops.OpOverload = exir_ops.edge.aten.view_copy.default
        self.squeeze_op: torch._ops.OpOverload = exir_ops.edge.aten.squeeze_copy.dims
        self.unsqueeze_op: torch._ops.OpOverload = (
            exir_ops.edge.aten.unsqueeze_copy.default
        )

    def is_node_target(
        self, node: torch.fx.Node, target: torch._ops.OperatorBase
    ) -> bool:
        return node.op == "call_function" and node.target == target

    def find_squeeze_dims(
        self,
        input_shape: List[int],
        view_shape: List[int],
    ) -> Optional[List[int]]:
        # view_shape should be a subset of input_shape
        if len(input_shape) <= len(view_shape):
            return None

        # check that all dims are equal except the removed dims
        i = 0
        j = 0
        idx = []
        while i < len(input_shape):
            if input_shape[i] != view_shape[j]:
                if input_shape[i] == 1:
                    idx.append(i)
                    j -= 1
                    # continue to check remaining dims are equal
                else:
                    return None
            i += 1
            j += 1
        return idx

    def find_unsqueeze_dim(
        self,
        input_shape: List[int],
        view_shape: List[int],
    ) -> Optional[int]:
        # unsqueeze should increase the length of input_shape by 1
        if len(view_shape) - len(input_shape) != 1:
            return None

        # check that all dims are equal except the added dim
        i = 0
        j = 0
        idx = -1
        while j < len(view_shape):
            if input_shape[i] != view_shape[j]:
                if view_shape[j] == 1:
                    idx = j
                    i -= 1
                    # continue to check remaining dims are equal
                else:
                    return None
            i += 1
            j += 1
        return idx

    def replace_view_copy_node(
        self,
        graph_module: torch.fx.GraphModule,
        view_node: torch.fx.Node,
        op: torch._ops.OpOverload,
        arg: Union[List[int], int],
    ) -> None:
        with graph_module.graph.inserting_before(view_node):
            new_node = graph_module.graph.create_node(
                "call_function",
                op,
                (view_node.args[0], arg),
            )
            new_node.meta = view_node.meta
            view_node.replace_all_uses_with(new_node)
            graph_module.graph.erase_node(view_node)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if self.is_node_target(node, self.view_copy_op):
                input_node = node.args[0]
                input_shape = input_node.meta["val"].shape
                view_shape = node.args[1]
                squeeze_dims = self.find_squeeze_dims(input_shape, view_shape)
                if squeeze_dims:
                    self.replace_view_copy_node(
                        graph_module, node, self.squeeze_op, squeeze_dims
                    )
                    modified = True
                    continue
                unsqueeze_dim = self.find_unsqueeze_dim(input_shape, view_shape)
                if unsqueeze_dim:
                    self.replace_view_copy_node(
                        graph_module, node, self.unsqueeze_op, unsqueeze_dim
                    )
                    modified = True
                    continue

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
