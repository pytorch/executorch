# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm._passes.insert_table_ops import TableOps
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.exir.dialects._ops import ops as exir_ops


class RemovePermutesAroundElementwiseTosaOps(RemovePermutesAroundElementwiseOps):
    permutable_ops = {
        *RemovePermutesAroundElementwiseOps.permutable_ops,
        *TableOps.unary_table_ops.keys(),
        *TableOps.special_table_ops,
        exir_ops.backend.tosa.RESCALE.default,
        exir_ops.backend.tosa.TABLE.default,
    }

    def permute_subgraph(self, subgraph):
        # Original function will always permute constant nodes which is wrong for table ops
        # Remove constant tosa.TABLE edges before running full function
        new_constant_edges_in = set()
        for const_node, user_node in subgraph.constant_edges_in:
            if user_node.target == exir_ops.backend.tosa.TABLE.default:
                continue
            else:
                new_constant_edges_in.add((const_node, user_node))

        subgraph.constant_edges_in = new_constant_edges_in
        super().permute_subgraph(subgraph)
