# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import logging

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from .arm_pass_utils import create_node, get_first_fake_tensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class InsertCastForOpsWithInt64InputPass(ExportPass):

    aten_ops = (
        torch.ops.aten.embedding.default,
        torch.ops.aten.slice_copy.Tensor,
    )
    edge_ops = (
        exir_ops.edge.aten.embedding.default,
        exir_ops.edge.aten.slice_copy.Tensor,
    )

    def get_decomposition(self, op):
        if op in self.edge_ops:
            return exir_ops.edge.aten._to_copy.default

        if op in self.aten_ops:
            return torch.ops.aten._to_copy.default

        raise RuntimeError(
            f"[{self.__class__.__name__}] Can't get decomposition for op {op}"
        )

    def _check_aten_embedding_within_int32(self, weights, indices, node: torch.fx.Node):
        weights_shape = get_first_fake_tensor(weights).shape
        vocab_size = weights_shape[0]

        # Essentially output = weight[indices] which means 0 <= indices[i] < vocab_size
        # So should be good if vocab size or number embeddings is below max int32
        if vocab_size >= torch.iinfo(torch.int32).max:
            logger.warning(
                f"[{node.name}] has size ({vocab_size}) that exceeds int32 limit,"
                "so aten.embedding will not be lowered to TOSA."
            )
            return False

        return True

    def _insert_int32_cast_before_node(self, graph, node, original_input):
        to_copy_op = self.get_decomposition(node.target)
        with graph.inserting_before(node):
            cast_before = create_node(
                graph,
                to_copy_op,
                args=(original_input,),
                kwargs={
                    "dtype": torch.int32,
                    "memory_format": torch.preserve_format,
                },
            )
            node.replace_input_with(original_input, cast_before)

    def call(self, graph_module):
        graph = graph_module.graph
        modified_graph = False

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target not in self.aten_ops + self.edge_ops:
                continue

            args = node.args

            if node.target in (
                exir_ops.edge.aten.embedding.default,
                torch.ops.aten.embedding.default,
            ):
                weights = args[0]
                indices = args[1]
                if self._check_aten_embedding_within_int32(weights, indices, node):
                    self._insert_int32_cast_before_node(graph, node, indices)
                    modified_graph = True

            elif node.target in (
                exir_ops.edge.aten.slice_copy.Tensor,
                torch.ops.aten.slice_copy.Tensor,
            ):
                # MLETORCH-829: Add range check for slice_copy
                input_tensor = args[0]
                fake_tensor = input_tensor.meta["val"]
                if fake_tensor.dtype != torch.int64:
                    continue

                self._insert_int32_cast_before_node(graph, node, input_tensor)
                modified_graph = True

        if modified_graph:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified_graph)
