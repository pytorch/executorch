# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast, List, Optional, Tuple

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    get_arg,
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload


class MovePermuteAfterConcat(RemoveOrReplacePassInterface):
    """Move matching single-use permutes after a cat.

    For example,

        cat(
            [
                permute(x[80, 16, 32], [1, 2, 0]),  # [16, 32, 80]
                permute(y[1, 16, 32], [1, 2, 0]),   # [16, 32, 1]
            ],
            dim=2,
        )  # [16, 32, 81]

    becomes

        permute(cat([x, y], dim=0), [1, 2, 0])
                # cat: [81, 16, 32], output: [16, 32, 81]
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.cat.default]

    @staticmethod
    def _match_inputs(
        cat_node: torch.fx.Node,
    ) -> Optional[Tuple[List[torch.fx.Node], Tuple[int, ...]]]:
        inputs = get_arg(
            cat_node,
            "tensors",
            # pyre-ignore[6]
            list[torch.fx.Node] | tuple[torch.fx.Node, ...],
        )
        # Moving one reused permute after cat makes it process the larger output.
        if inputs and all(inp is inputs[0] for inp in inputs):
            return None

        unpermuted_inputs: List[torch.fx.Node] = []
        common_permutation: Optional[Tuple[int, ...]] = None
        for permute_node in inputs:
            if (
                permute_node.target != exir_ops.edge.aten.permute_copy.default
                or len(permute_node.users) != 1
            ):
                return None

            permute_input = cast(torch.fx.Node, permute_node.args[0])
            dims = get_arg(
                permute_node,
                "dims",
                # pyre-ignore[6]
                list[int] | tuple[int, ...],
            )
            rank = len(dims)
            permutation = tuple(dim % rank for dim in dims)
            if common_permutation is None:
                common_permutation = permutation
            elif permutation != common_permutation:
                return None
            unpermuted_inputs.append(permute_input)

        return unpermuted_inputs, cast(Tuple[int, ...], common_permutation)

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        match = self._match_inputs(node)
        if match is None:
            return False
        inputs, permutation = match

        output_cat_dim = get_arg(node, "dim", int) % len(permutation)
        input_cat_dim = permutation[output_cat_dim]
        graph = node.graph

        with graph.inserting_before(node):
            new_cat = graph.call_function(
                exir_ops.edge.aten.cat.default,
                args=(inputs, input_cat_dim),
            )
            new_cat.meta["val"] = exir_ops.edge.aten.cat.default(
                [inp.meta["val"] for inp in inputs], input_cat_dim
            )
            new_permute = graph.call_function(
                exir_ops.edge.aten.permute_copy.default,
                args=(new_cat, list(permutation)),
            )
            new_permute.meta = node.meta.copy()

        node.replace_all_uses_with(new_permute)
        return True
