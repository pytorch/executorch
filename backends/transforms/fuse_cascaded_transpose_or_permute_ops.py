# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from executorch.backends.transforms.permute_pass_utils import (
    get_arg,
    get_permuted_dims,
    get_transposed_dims,
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch.fx import Node


class FuseCascadedTransposeOrPermuteOps(RemoveOrReplacePassInterface):
    """
    Fuse a chain of transpose and permute ops into a single permute or a no-op.
    Handles branches and chains permutes.
    """

    transpose_or_permute_target = {
        exir_ops.edge.aten.transpose_copy.int,
        exir_ops.edge.aten.permute_copy.default,
    }

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return list(self.transpose_or_permute_target)

    def maybe_remove_or_replace(self, node: Node) -> bool:
        # Fuse with the parent node if it's also a permute or a transpose. Since the
        # pass interface traverses all ops in order the pass will properly fuse a chain
        # of permutes.
        parent_node = get_arg(node, "input", Node)
        if parent_node.target not in self.transpose_or_permute_target:
            return False
        input_of_parent = get_arg(parent_node, "input", Node)

        # Compute combined effect of permutes.
        dims = list(range(node.meta["val"].ndim))

        if parent_node.target == exir_ops.edge.aten.transpose_copy.int:
            dims = get_transposed_dims(parent_node, dims)
        else:
            dims = get_permuted_dims(parent_node, dims)

        if node.target == exir_ops.edge.aten.transpose_copy.int:
            dims = get_transposed_dims(node, dims)
        else:
            dims = get_permuted_dims(node, dims)

        # If combined effect is identity replace the node with input.
        if dims == sorted(dims):
            node.replace_all_uses_with(input_of_parent)
        else:
            with node.graph.inserting_before(node):
                new_permute = node.graph.call_function(
                    exir_ops.edge.aten.permute_copy.default,
                    args=(input_of_parent, dims),
                )
                new_permute.meta = node.meta
            node.replace_all_uses_with(new_permute)

        return True
