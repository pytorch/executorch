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


class RemoveZeroSizedCatArgsPass(RemoveOrReplacePassInterface):
    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.cat.default]

    def maybe_remove_or_replace(self, node: Node) -> bool:
        # Get the cat inputs (first argument is a list of tensors)
        cat_inputs_arg = node.args[0]

        # Assert that cat_inputs_arg is iterable
        assert isinstance(cat_inputs_arg, (list, tuple)), (
            "cat_inputs_arg must be a sequence type"
        )

        # Filter out zero-sized tensors
        cat_inputs: list[Node] = []
        for arg in cat_inputs_arg:
            if isinstance(arg, Node) and arg.meta.get("val") is not None:
                if arg.meta["val"].numel() > 0:
                    cat_inputs.append(arg)

        # If all tensors were empty, create a full op with the right shape
        if not cat_inputs:
            empty_shape = node.meta["val"].shape
            dtype = node.meta["val"].dtype
            # Create a new full node
            with node.graph.inserting_before(node):
                full_node = node.graph.call_function(
                    exir_ops.edge.aten.full.default,
                    args=(tuple(empty_shape), 0),
                    kwargs={"dtype": dtype},
                )
                full_node.meta = node.meta.copy()
            node.replace_all_uses_with(full_node)
            return True

        # If only one tensor remains, replace with it
        if len(cat_inputs) == 1:
            node.replace_all_uses_with(cat_inputs[0])
            return True

        # If the number of inputs changed, update the cat args
        if len(cat_inputs) < len(cat_inputs_arg):
            # Update the first argument with filtered inputs
            new_args = list(node.args)
            new_args[0] = cat_inputs
            node.args = tuple(new_args)
            return True

        # No changes needed
        return False
