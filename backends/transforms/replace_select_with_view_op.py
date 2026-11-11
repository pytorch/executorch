# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload


class ReplaceSelectWithViewOpPass(RemoveOrReplacePassInterface):
    """
    If the size along the select dim is 1, then the select op can be replaced
    by view op.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.select_copy.int]

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        # Get the input tensor and shapes
        in_tensor_node = node.args[0]
        assert isinstance(in_tensor_node, torch.fx.Node)
        in_shape = in_tensor_node.meta["val"].shape
        out_shape = node.meta["val"].shape

        # Get the select dimension
        select_dim = node.args[1]
        assert isinstance(select_dim, int)
        select_dim = select_dim if select_dim >= 0 else select_dim + len(in_shape)

        if in_shape[select_dim] == 1:
            # Replace with view op with the new shape
            with node.graph.inserting_before(node):
                new_node = node.graph.call_function(
                    exir_ops.edge.aten.view_copy.default,
                    args=(node.args[0], list(out_shape)),
                )
                # Important to copy metadata
                new_node.meta = node.meta
            node.replace_all_uses_with(new_node)
            return True

        return False
