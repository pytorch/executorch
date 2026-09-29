# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult

from .utils import copy_meta


class DecomposeSelectScatter(ExportPass):
    """
    Decompose select_scatter into unsqueeze + slice_scatter.

    select_scatter(input, src, dim, index) replaces a single index along the given dimension.
    If input has shape [m, n, p] and dim=1, then src must have shape [m, p] (the selected dimension is removed).
    slice_scatter operates on a sliced view where the dimension is preserved.
    When slicing a single index, the target region has shape [m, 1, p].

    Therefore, src must be unsqueezed along dim (from [m, p] to [m, 1, p]) to match the slice shape.
    So, the equivalence is:
        select_scatter(input, src, dim, index) == slice_scatter(input, src.unsqueeze(dim), dim, index, index+1, 1)
    """

    def __init__(self):
        super(DecomposeSelectScatter, self).__init__()
        self.select_scatter_targets = {
            torch.ops.aten.select_scatter.default,
            exir_ops.edge.aten.select_scatter.default,
        }

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph

        for node in list(graph.nodes):
            if (
                node.op == "call_function"
                and node.target in self.select_scatter_targets
            ):
                input_node = node.args[0]
                src_node = node.args[1]
                dim = node.args[2]
                index = node.args[3]

                # Normalize negative index
                if index < 0:
                    size = input_node.meta["val"].shape[dim]
                    index = index + size

                is_edge = isinstance(node.target, EdgeOpOverload)
                meta = node.meta

                unsqueeze_op = (
                    exir_ops.edge.aten.unsqueeze_copy.default
                    if is_edge
                    else torch.ops.aten.unsqueeze.default
                )
                slice_scatter_op = (
                    exir_ops.edge.aten.slice_scatter.default
                    if is_edge
                    else torch.ops.aten.slice_scatter.default
                )

                with graph.inserting_before(node):
                    # unsqueeze src along dim to restore the missing dimension
                    unsqueeze_node = graph.create_node(
                        "call_function", unsqueeze_op, (src_node, dim)
                    )
                    # Compute unsqueeze output shape for meta
                    src_val = src_node.meta.get("val", None)
                    if src_val is not None:
                        unsqueeze_val = src_val.unsqueeze(dim)
                        unsqueeze_node.meta = copy_meta(
                            meta,
                            callback=lambda m, val=unsqueeze_val: {**m, "val": val},
                        )
                    else:
                        unsqueeze_node.meta = copy_meta(meta)

                    slice_scatter_node = graph.create_node(
                        "call_function",
                        slice_scatter_op,
                        (input_node, unsqueeze_node, dim, index, index + 1, 1),
                    )
                    slice_scatter_node.meta = copy_meta(meta)

                for user in node.users.copy():
                    user.replace_input_with(node, slice_scatter_node)

        graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
