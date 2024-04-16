# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

import torch

from torch.fx.passes.infra.pass_base import PassBase, PassResult

_squeeze_copy_to_view_copy = {  # pyre-ignore
    torch.ops.aten.squeeze_copy.dims: torch.ops.aten.view_copy.default,
    torch.ops.aten.squeeze_copy.dim: torch.ops.aten.view_copy.default,
    torch.ops.aten.squeeze_copy.default: torch.ops.aten.view_copy.default,
}


def maybe_replace_squeeze_copy_with_view_copy(node: torch.fx.Node) -> bool:
    """
    Replace a squeeze_copy with a view_copy.

    Symbolic shapes are assumed to not be 0/1 in torch.export.export
    (https://pytorch.org/docs/stable/export.html#expressing-dynamism), but we
    still avoid converting squeeze ops to view ops when the dim is symbolic or
    the dim refers to a symbolic shape to avoid any unexpected behavior.
    """
    assert node.target in _squeeze_copy_to_view_copy

    base = node.args[0]
    size = base.meta["val"].shape  # pyre-ignore

    if len(node.args) == 1:
        # squeeze_copy.default
        dims = list(range(len(size)))
    else:
        # squeeze_copy.dim or squeeze_copy.dims
        dim_or_dims = node.args[1]
        dims = dim_or_dims
        if isinstance(dim_or_dims, int):
            dims = [dim_or_dims]

    # Check for unsupported dynamism
    for d in dims:  # pyre-ignore
        if not isinstance(d, int):
            return False
        if not isinstance(size[d], int):
            return False

    new_size = list(node.meta["val"].shape)
    node.target = _squeeze_copy_to_view_copy[node.target]
    node.args = (base, new_size)
    return True


class ReplaceSqueezeCopyWithViewCopyPass(PassBase):
    """
    This pass replaces squeeze_copy nodes with view_copy nodes when the conversion
    can be done statically ahead of time, i.e., the number of arguments in the view_copy
    does not depend on symbolic shapes.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        n_replaced = 0
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if (
                    node.op == "call_function"
                    and node.target in _squeeze_copy_to_view_copy
                ):
                    if maybe_replace_squeeze_copy_with_view_copy(node):
                        n_replaced += 1

            module.recompile()

        logging.debug(f"Replaced {n_replaced} squeeze_copy nodes with view_copy nodes.")

        return PassResult(graph_module, n_replaced > 0)
