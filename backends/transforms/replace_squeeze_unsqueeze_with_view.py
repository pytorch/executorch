# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.backends.transforms.permute_pass_utils import (
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload


class ReplaceSqueezeAndUnsqueezeWithViewPass(RemoveOrReplacePassInterface):
    """
    When the shape is static, replace squeeze_copy and unsqueeze_copy ops with
    view_copy op.

    Canonicalising the rank-changing shape ops down to a single ``view_copy``
    form lets downstream layout passes reason about one operator instead of
    three. Run this before any pass that needs to move a permutation across a
    rank change.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.aten.squeeze_copy.default,
            exir_ops.edge.aten.squeeze_copy.dim,
            exir_ops.edge.aten.squeeze_copy.dims,
            exir_ops.edge.aten.unsqueeze_copy.default,
        ]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        out_shape = node.meta["val"].shape

        # Bail out if any dim is not an int (dynamic shape)
        for dim in list(out_shape):
            if not isinstance(dim, int):
                return False

        with node.graph.inserting_before(node):
            new_node = node.graph.call_function(
                exir_ops.edge.aten.view_copy.default,
                args=(node.args[0], list(out_shape)),
            )
            # Do not remove the metadata copy!
            new_node.meta = node.meta
        node.replace_all_uses_with(new_node)
        return True
