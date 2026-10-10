# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from executorch.backends.transforms.permute_pass_utils import (
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch.fx.node import Node


class RemoveNopSliceOrViewOpPass(RemoveOrReplacePassInterface):
    """
    Remove slice ops that are more like views, and view ops that do not change the shape
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.aten.slice_copy.Tensor,
            exir_ops.edge.aten.view_copy.default,
        ]

    def maybe_remove_or_replace(self, node: Node) -> bool:
        changed = False
        input_node = node.args[0]
        assert isinstance(input_node, Node)
        if input_node.meta["val"].shape == node.meta["val"].shape:
            node.replace_all_uses_with(input_node)
            changed = True

        return changed
