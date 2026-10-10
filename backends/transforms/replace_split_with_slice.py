# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import operator
from typing import Optional

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload


class ReplaceSplitWithSlicePass(RemoveOrReplacePassInterface):
    """
    split_with_sizes() delegates to slice() op, so perform this replacement here.
    This avoids the expense of delegation from ATen.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.split_with_sizes_copy.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # All the users of this split_with_sizes op must be getitem ops
        if any(user.target != operator.getitem for user in node.users):
            return False

        # Get the slice dim and extent for each split
        slice_ops = self._get_split_sizes(node)
        if slice_ops is None:
            return False

        graph = node.graph

        # Go over each getitem user, and replace it with slice op
        for user in list(node.users.keys()):
            assert user.target == operator.getitem
            item_idx = int(user.args[1])
            assert item_idx < len(slice_ops)
            cur_slice = slice_ops[item_idx]
            with graph.inserting_before(user):
                cur_slice_node = graph.call_function(
                    exir_ops.edge.aten.slice_copy.Tensor,
                    (node.args[0], cur_slice[0], cur_slice[1], cur_slice[2], 1),
                )
                # Metadata copy important
                cur_slice_node.meta = user.meta
            user.replace_all_uses_with(cur_slice_node)

        # Return True to indicate the split node should be removed
        return True

    def _get_split_sizes(self, node: torch.fx.Node) -> Optional[list[tuple[int, ...]]]:
        """For split_with_sizes, return the slice dim and extent for each split."""
        # Parse the args of the split_with_sizes op
        tensor_arg, split_sizes = node.args[0:2]
        assert isinstance(tensor_arg, torch.fx.Node)

        # Get shape from node metadata
        val = tensor_arg.meta.get("val")
        if val is None:
            return None
        in_shape = val.shape

        split_dim = 0 if len(node.args) < 3 else node.args[2]

        # Canonicalize the split dimension
        assert isinstance(split_dim, int)
        split_dim = split_dim if split_dim >= 0 else len(in_shape) + split_dim

        # Create the slice op args corresponding to each split
        slice_ops = []
        split_start = 0
        assert isinstance(split_sizes, list)
        for split_size in split_sizes:
            split_end = split_start + split_size
            slice_args = (split_dim, split_start, split_end)
            slice_ops.append(slice_args)
            split_start = split_end

        return slice_ops
