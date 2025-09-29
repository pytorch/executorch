# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import logging
from math import prod
from typing import Set, Type

import torch
from executorch.backends.transforms.fuse_view_copy import FuseViewCopyTransform
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from .arm_pass_utils import create_node, get_first_fake_tensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class DecomposeEmbeddingPass(ExportPass):
    """
    This pass decomposes embedding into index_select.

    Example:
          o = embedding(w, i)
    Becomes:
          i = view_copy(i)  # flatten indices
          o = index_select(w, i)
          o = view_copy(o)  # reshape back output
    Note:
         i = indices is expected to be int32 before this pass
    """

    _passes_required_after: Set[Type[ExportPass]] = {FuseViewCopyTransform}

    aten_ops = (torch.ops.aten.embedding.default,)
    edge_ops = (exir_ops.edge.aten.embedding.default,)

    def get_decomposition(self, op):
        if op in self.aten_ops:
            return (
                torch.ops.aten.view_copy.default,
                torch.ops.aten.index_select.default,
            )

        if op in self.edge_ops:
            return (
                exir_ops.edge.aten.view_copy.default,
                exir_ops.edge.aten.index_select.default,
            )
        raise RuntimeError(
            f"[{self.__class__.__name__}] Can't get decomposition for op {op}"
        )

    def call(self, graph_module):
        graph = graph_module.graph
        modified_graph = False

        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target not in self.aten_ops + self.edge_ops:
                continue

            args = node.args

            weights = args[0]
            indices = args[1]

            weights_shape = get_first_fake_tensor(weights).shape
            indices_shape = get_first_fake_tensor(indices).shape

            output_shape = torch.Size(list(indices_shape) + [weights_shape[1]])
            if output_shape != get_first_fake_tensor(node).shape:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Unexpected output shape mismatch {output_shape} "
                    "!= {get_first_fake_tensor(node).shape}"
                )

            view_copy_op, index_select_op = self.get_decomposition(node.target)

            with graph.inserting_before(node):
                reshaped_indices = [prod(list(indices_shape))]
                flattened_indices = create_node(
                    graph=graph,
                    op_target=view_copy_op,
                    args=(indices, reshaped_indices),
                )
                node.replace_input_with(indices, flattened_indices)

                index_select = create_node(
                    graph=graph,
                    op_target=index_select_op,
                    args=(weights, 0, flattened_indices),
                )
                node.replace_all_uses_with(index_select)
                graph.erase_node(node)

            with graph.inserting_after(index_select):
                restored_output = create_node(
                    graph,
                    view_copy_op,
                )
                restored_output.args = (
                    index_select,
                    output_shape,
                )
                original_users = [
                    user for user in index_select.users if user != restored_output
                ]
                for user in original_users:
                    user.replace_input_with(index_select, restored_output)

            modified_graph = True

        if modified_graph:
            graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified_graph)
