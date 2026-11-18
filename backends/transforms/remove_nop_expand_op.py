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


class RemoveNopExpandOpPass(RemoveOrReplacePassInterface):
    """
    For an expand op, if the operator shape matches the expand shape, then the
    expand is a nop.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.aten.expand_copy.default,
            exir_ops.edge.aten.expand.default,
        ]

    def maybe_remove_or_replace(self, node: Node) -> bool:
        input_node = node.args[0]
        assert isinstance(input_node, Node)
        if input_node.meta["val"].shape == node.meta["val"].shape:
            node.replace_all_uses_with(input_node)
            return True
        return False
