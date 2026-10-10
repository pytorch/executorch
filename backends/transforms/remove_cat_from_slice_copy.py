# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional, Sequence

from executorch.backends.transforms.permute_pass_utils import (
    get_arg,
    RemoveOrReplacePassInterface,
    set_arg,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch.fx.node import Node


class RemoveCatFromSliceCopyPass(RemoveOrReplacePassInterface):
    """
    Simplifies cat->slice_copy chains where one of the cat inputs can be directly passed
    to the slice_copy.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.slice_copy.Tensor]

    def maybe_remove_or_replace(self, node: Node) -> bool:
        cat_node = get_arg(node, "input", Node)
        slice_dim = get_arg(node, "dim", int)
        start_idx = get_arg(node, "start", Optional[int])
        end_idx = get_arg(node, "end", Optional[int])
        step = get_arg(node, "step", int)

        if cat_node.target != exir_ops.edge.aten.cat.default or step != 1:
            return False

        # Make sure cat and slice happens on the same dimension.
        cat_dim = get_arg(cat_node, "dim", int)
        if cat_dim != slice_dim:
            return False

        # Canonicalize slice indices.
        cat_output_shape = cat_node.meta["val"].shape
        if start_idx is None:
            start_idx = 0
        elif start_idx < 0:
            start_idx += cat_output_shape[cat_dim]
        if end_idx is None or end_idx > cat_output_shape[cat_dim]:
            end_idx = cat_output_shape[cat_dim]
        elif end_idx < 0:
            end_idx += cat_output_shape[cat_dim]

        offset = 0
        for cat_input_node in get_arg(cat_node, "tensors", Sequence[Node]):
            cat_input_shape = cat_input_node.meta["val"].shape

            # Check if the slice range overlaps with the cat input range.
            if offset <= start_idx and end_idx <= offset + cat_input_shape[cat_dim]:
                node.replace_input_with(cat_node, cat_input_node)
                set_arg(node, "start", start_idx - offset)
                set_arg(node, "end", end_idx - offset)
                return True

            offset += cat_input_shape[cat_dim]

        return False
