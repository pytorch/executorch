# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)
from executorch.backends.arm._passes.convert_squeezes_to_view import (
    ConvertSqueezesToViewPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposeSelectPass(ExportPass):
    """
    This pass decomposes select into slice + squeeze to ensure that Aten and TOSA outputs has the same rank (input rank -1)
    """

    _passes_required_after: Set[Type[ExportPass]] = {ConvertSqueezesToViewPass}

    def call(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:

            if node.op != "call_function":
                continue

            if node.target in (
                exir_ops.edge.aten.select.int,
                exir_ops.edge.aten.select_copy.int,
            ):
                slice_op = exir_ops.edge.aten.slice_copy.Tensor
                squeeze_op = exir_ops.edge.aten.squeeze_copy.dims
            else:
                continue

            input_node, dim, index = node.args

            input_tensor = get_first_fake_tensor(input_node)
            rank = len(input_tensor.size())
            shape = input_tensor.shape
            dim = dim % rank if dim < 0 else dim
            index = index % shape[dim] if index < 0 else index

            with graph_module.graph.inserting_before(node):
                slice_node = create_node(
                    graph_module.graph, slice_op, (input_node, dim, index, index + 1)
                )
                squeeze_node = create_node(
                    graph_module.graph, squeeze_op, (slice_node, [dim]), from_node=node
                )

            node.replace_all_uses_with(squeeze_node)
            graph_module.graph.erase_node(node)

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
