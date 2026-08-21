# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm._passes.insert_table_ops import TableOps
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops


class RemovePermutesAroundElementwiseTosaOps(RemovePermutesAroundElementwiseOps):
    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__(
            extra_permutable_ops={
                *TableOps.unary_table_ops.keys(),
                *TableOps.special_table_ops,
                exir_ops.backend.tosa.RESCALE.default,
                exir_ops.backend.tosa.TABLE.default,
            }
        )
        self.exported_program = exported_program
        # Precompute parameter/buffer/lifted-constant placeholder names once.
        # is_param_node() rebuilds these graph_signature maps on every call, so
        # calling it per node made the visit() walk O(nodes * inputs).
        gs = exported_program.graph_signature
        self._constant_input_names: set[str] = (
            set(gs.inputs_to_parameters)
            | set(gs.inputs_to_buffers)
            | set(gs.inputs_to_lifted_tensor_constants)
        )

    def _is_constant(self, node: torch.fx.Node) -> bool:
        # get_attr nodes are handled by super()._is_constant; set membership
        # here is equivalent to is_param_node for placeholder inputs.
        return super()._is_constant(node) or node.name in self._constant_input_names

    def permute_subgraph(self, subgraph) -> bool:
        # TABLE lookup inputs are already tied to the table layout.
        new_constant_edges_in = set()
        for const_node, user_node in subgraph.constant_edges_in:
            if user_node.target == exir_ops.backend.tosa.TABLE.default:
                continue
            new_constant_edges_in.add((const_node, user_node))

        subgraph.constant_edges_in = new_constant_edges_in
        return super().permute_subgraph(subgraph)
