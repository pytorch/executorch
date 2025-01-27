# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from executorch.backends.xnnpack.utils.utils import get_input_node
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass, PassResult

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ExpandToViewPass(ExportPass):
    """
    Torch expand_copy can be used as an altenative to unsqueeze. This pass replaces expand_copy nodes
    that only add one or more singleton dimensions.


    Example:
    Before Pass:
        expand: "f32" = torch.ops.aten.expand_copy.default(x, (1, -1));

    After Pass:
        view: "f32" = torch.ops.aten.view_copy.default(x, (1, -1));
    """

    @staticmethod
    def can_transform_expand_node(node: torch.fx.Node) -> bool:
        # The node can be converted to a view if the expand only inserts singleton dimensions and
        # does not modify any existing dimensions.
        in_shape = get_input_node(node, 0).meta["val"].shape
        out_shape = node.meta["val"].shape

        i = 0  # in-shape index
        j = 0  # out-shape index
        while j < len(out_shape):
            if i >= len(in_shape):  # Shape mismatch
                return False
            elif in_shape[i] == out_shape[j]:  # Dims match
                i += 1
                j += 1
            elif out_shape[j] == 1:  # Inserted singleton dim
                j += 1
            else:  # Dim mismatch (in_shape[i] != out_shape[i])
                return False

        return True

    def call(self, graph_module: torch.fx.GraphModule):
        gm = graph_module
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == exir_ops.edge.aten.expand_copy.default
                and ExpandToViewPass.can_transform_expand_node(node)
            ):
                with gm.graph.inserting_after(node):
                    view_node = gm.graph.create_node(
                        "call_function",
                        target=exir_ops.edge.aten.view_copy.default,
                        args=(node.args[0], node.args[1]),
                        kwargs=node.kwargs,
                    )

                    node.replace_all_uses_with(view_node)
                    gm.graph.erase_node(node)

        gm.recompile()
        new_gm = super().call(gm).graph_module
        return PassResult(new_gm, True)
