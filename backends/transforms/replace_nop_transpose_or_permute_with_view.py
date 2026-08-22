# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast, Sequence

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload


class ReplaceNopTransposeOrPermuteWithViewPass(RemoveOrReplacePassInterface):
    """
    If the transpose/permute op does not change the byte order (e.g.,
    transpose/permute from Nx1xHxW to NxHx1xW), then it can be replaced
    by view op.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.aten.transpose_copy.int,
            exir_ops.edge.aten.permute_copy.default,
        ]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Get the input tensor and shape
        in_tensor_node = node.args[0]
        assert isinstance(in_tensor_node, torch.fx.Node)
        in_shape = in_tensor_node.meta["val"].shape
        # Get the output tensor shape
        out_shape = node.meta["val"].shape

        if node.target == exir_ops.edge.aten.transpose_copy.int:
            # Get the two dims to be transposed
            dim0 = cast(int, node.args[1])
            dim1 = cast(int, node.args[2])
            dim0 = dim0 if dim0 >= 0 else len(in_shape) + dim0
            dim1 = dim1 if dim1 >= 0 else len(in_shape) + dim1
            # We can eliminate transpose if (a) the size at dim0 and dim1 is 1;
            # (b) the size at dim0 or dim1 is 1, and dim0 and dim1 are consecutive.
            both_one = in_shape[dim0] == 1 and in_shape[dim1] == 1
            either_one_and_consecutive = abs(dim0 - dim1) == 1 and (
                in_shape[dim0] == 1 or in_shape[dim1] == 1
            )
            if both_one or either_one_and_consecutive:
                with node.graph.inserting_before(node):
                    new_node = node.graph.call_function(
                        exir_ops.edge.aten.view_copy.default,
                        args=(in_tensor_node, list(out_shape)),
                    )
                    new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                return True

        elif node.target == exir_ops.edge.aten.permute_copy.default:
            old_dims = list(range(len(in_shape)))
            new_dims = cast(Sequence[int], node.args[1])
            # If the permute does not change anything, return the input as output.
            if old_dims == list(new_dims):
                node.replace_all_uses_with(in_tensor_node)
                return True
            # Get the old dim order, and the permuted dim order for all dims that
            # are not 1.
            old_order = [
                dim for dim, shape_dim in zip(old_dims, in_shape) if shape_dim != 1
            ]
            new_order = [
                dim for dim, shape_dim in zip(new_dims, out_shape) if shape_dim != 1
            ]
            # If the byte ordering for non-unit dims is unchanged, this is a nop.
            if old_order == new_order:
                with node.graph.inserting_before(node):
                    new_node = node.graph.call_function(
                        exir_ops.edge.aten.view_copy.default,
                        args=(in_tensor_node, list(out_shape)),
                    )
                    new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                return True

        return False
