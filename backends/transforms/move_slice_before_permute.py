# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from math import prod
from typing import cast

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    get_arg,
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload


class MoveSliceBeforePermutePass(RemoveOrReplacePassInterface):
    """Move slice_copy ops before permute_copy to reduce permute data volume.

    Rewrites permute(input, perm) -> slice(dim=D) into
    slice(input, dim=perm[D]) -> permute(sliced, perm), so the permute
    operates on a smaller tensor.

    Scans slice nodes and matches upstream permutes. This also handles
    chained cases (permute -> slice -> slice) in one pass: each slice
    independently checks its input for a permute.

    Cost model: dim-0 slices are nop-eligible (zero-copy pointer offset
    after MakeSliceAndCatDimOutermostPass).  Moving such a slice loses the
    nop, so we only move it when the permute savings outweigh the nop loss,
    i.e. when the slice removes more than half the data (full > 2 * sliced).
    Non-dim-0 slices have no nop opportunity, so any permute savings is
    pure win.
    """

    STRIDED_SLICE_COST_FACTOR: int = 2

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.slice_copy.Tensor]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        permute_node = get_arg(node, "input", torch.fx.Node)
        if permute_node.target != exir_ops.edge.aten.permute_copy.default:
            return False

        if len(permute_node.users) != 1:
            return False

        perm = cast(list[int], permute_node.args[1])
        permute_input = permute_node.args[0]
        assert isinstance(permute_input, torch.fx.Node)

        slice_dim = get_arg(node, "dim", int)

        full_size = prod(permute_node.meta["val"].shape)
        sliced_size = prod(node.meta["val"].shape)
        if slice_dim == 0 and full_size <= self.STRIDED_SLICE_COST_FACTOR * sliced_size:
            return False

        new_dim = perm[slice_dim]
        graph = node.graph

        with graph.inserting_before(permute_node):
            new_slice_args = (
                permute_input,
                new_dim,
                get_arg(node, "start"),
                get_arg(node, "end"),
                get_arg(node, "step", int),
            )
            new_slice = graph.create_node(
                "call_function",
                exir_ops.edge.aten.slice_copy.Tensor,
                args=new_slice_args,
            )
            new_slice.meta["val"] = exir_ops.edge.aten.slice_copy.Tensor(
                permute_input.meta["val"], *new_slice_args[1:]
            )
            new_permute = graph.create_node(
                "call_function",
                exir_ops.edge.aten.permute_copy.default,
                args=(new_slice, perm),
            )
            new_permute.meta["val"] = exir_ops.edge.aten.permute_copy.default(
                new_slice.meta["val"], perm
            )

        node.replace_all_uses_with(new_permute)
        return True
