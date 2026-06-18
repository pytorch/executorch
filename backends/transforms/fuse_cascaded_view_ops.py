# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast

import torch
from executorch.backends.transforms.permute_pass_utils import (
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload


class FuseCascadedViewOps(RemoveOrReplacePassInterface):
    """
    Fuse a cascaded chain of view ops
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.view_copy.default]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Check if the input to this view node is also a view node
        input_view = node.args[0]
        if not isinstance(input_view, torch.fx.Node):
            return False

        if (
            input_view.op != "call_function"
            or input_view.target != exir_ops.edge.aten.view_copy.default
        ):
            return False

        # Replace the input of this view node with the input of the cascaded view
        # This effectively "skips" the intermediate view node
        node.replace_input_with(input_view, cast(torch.fx.Node, input_view.args[0]))
        return True
