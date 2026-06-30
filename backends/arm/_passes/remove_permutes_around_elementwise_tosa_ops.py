# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm._passes.arm_pass_utils import is_param_node
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

    def _is_constant(self, node: torch.fx.Node) -> bool:
        # Override fragile string match check with exported program check
        return super()._is_constant(node) or is_param_node(self.exported_program, node)

    def permute_subgraph(self, subgraph) -> bool:
        # TABLE lookup inputs are already tied to the table layout.
        new_constant_edges_in = set()
        for const_node, user_node in subgraph.constant_edges_in:
            if user_node.target == exir_ops.backend.tosa.TABLE.default:
                continue
            new_constant_edges_in.add((const_node, user_node))

        subgraph.constant_edges_in = new_constant_edges_in
        return super().permute_subgraph(subgraph)
