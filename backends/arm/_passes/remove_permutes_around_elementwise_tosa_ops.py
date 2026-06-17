# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm._passes.insert_table_ops import TableOps
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.exir.dialects._ops import ops as exir_ops


class RemovePermutesAroundElementwiseTosaOps(RemovePermutesAroundElementwiseOps):
    def __init__(self) -> None:
        super().__init__(
            extra_permutable_ops={
                *TableOps.unary_table_ops.keys(),
                *TableOps.special_table_ops,
                exir_ops.backend.tosa.RESCALE.default,
                exir_ops.backend.tosa.TABLE.default,
            }
        )

    @staticmethod
    def _is_scalar_constant_value(node):
        value = node.meta.get("val")
        return isinstance(value, torch.Tensor) and value.numel() == 1

    def _is_constant(self, node):
        if super()._is_constant(node):
            return True

        if node.op != "placeholder":
            return False

        target = str(node.target)
        if not target.endswith(("_fused_const", "_pre_computed")):
            return False

        return self._is_scalar_constant_value(node)

    def permute_subgraph(self, subgraph):
        # Original function will always permute constant nodes. Skip constants
        # where the compensating permute is either wrong or a no-op.
        new_constant_edges_in = set()
        for const_node, user_node in subgraph.constant_edges_in:
            if (
                user_node.target == exir_ops.backend.tosa.TABLE.default
                or self._is_scalar_constant_value(const_node)
            ):
                continue
            new_constant_edges_in.add((const_node, user_node))

        subgraph.constant_edges_in = new_constant_edges_in
        return super().permute_subgraph(subgraph)
