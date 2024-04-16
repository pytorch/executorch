# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

import torch

from torch.fx.passes.infra.pass_base import PassBase, PassResult

_unsqueeze_copy_to_view_copy = {  # pyre-ignore
    torch.ops.aten.unsqueeze_copy.default: torch.ops.aten.view_copy.default,
}


def maybe_replace_unsqueeze_copy_with_view_copy(node: torch.fx.Node) -> bool:
    """
    Replace an unsqueeze_copy with a view_copy.

    Symbolic shapes are assumed to not be 0/1 in torch.export.export
    (https://pytorch.org/docs/stable/export.html#expressing-dynamism), but we
    still avoid converting unsqueeze ops to view ops when the dim is symbolic or
    the dim refers to a symbolic shape to avoid any unexpected behavior.
    """
    assert node.target in _unsqueeze_copy_to_view_copy

    base = node.args[0]
    dim = node.args[1]

    # Check for unsupported dynamism
    if not isinstance(dim, int):
        return False

    new_size = list(node.meta["val"].shape)
    node.target = _unsqueeze_copy_to_view_copy[node.target]
    node.args = (base, new_size)
    return True


class ReplaceUnsqueezeCopyWithViewCopyPass(PassBase):
    """
    This pass replaces unsqueeze_copy nodes with view_copy nodes when the conversion
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
                    and node.target in _unsqueeze_copy_to_view_copy
                ):
                    if maybe_replace_unsqueeze_copy_with_view_copy(node):
                        n_replaced += 1

            module.recompile()

        logging.debug(
            f"Replaced {n_replaced} unsqueeze_copy nodes with view_copy nodes."
        )

        return PassResult(graph_module, n_replaced > 0)
