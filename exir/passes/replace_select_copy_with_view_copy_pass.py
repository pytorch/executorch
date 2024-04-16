# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

import torch

from torch.fx.passes.infra.pass_base import PassBase, PassResult

_select_copy_to_view_copy = {  # pyre-ignore
    torch.ops.aten.select_copy.int: torch.ops.aten.view_copy.default,
}


def maybe_replace_select_copy_with_view_copy(node: torch.fx.Node) -> bool:
    """
    Replace an select_copy with a view_copy if the size in the
    select dim is 1.
    """
    assert node.target in _select_copy_to_view_copy

    base = node.args[0]
    in_size = base.meta["val"].shape  # pyre-ignore
    out_size = list(node.meta["val"].shape)
    dim = node.args[1] if node.args[1] >= 0 else node.args[1] + len(in_size)  # pyre-ignore
    index = node.args[2]

    if in_size[dim] != 1:
        return False
    assert index == 0

    node.target = _select_copy_to_view_copy[node.target]
    node.args = (base, out_size)
    return True


class ReplaceSelectCopyWithViewCopyPass(PassBase):
    """
    This pass replaces select_copy with a view_copy if the size in the
    select dim is 1.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        n_replaced = 0
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if (
                    node.op == "call_function"
                    and node.target in _select_copy_to_view_copy
                ):
                    if maybe_replace_select_copy_with_view_copy(node):
                        n_replaced += 1

            module.recompile()

        logging.debug(f"Replaced {n_replaced} select_copy nodes with view_copy nodes.")

        return PassResult(graph_module, n_replaced > 0)
