# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


# This file contains all the functions that simplify args of an op

import sys
from typing import cast, Optional

import torch
from executorch.backends.cadence.aot.pass_utils import (
    CadencePassAttribute,
    register_cadence_pass,
    RemoveOrReplacePassInterface,
)
from executorch.backends.cadence.aot.utils import rebind
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import Node


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class SimplifySliceOpPass(RemoveOrReplacePassInterface):
    """
    Simplify the start and end indices of slice and slice_scatter ops.
    """

    def adjust_slice_range(
        self,
        length: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: int = 1,
    ) -> tuple[int, int]:
        # Get the start index and end index
        start_val = start if start is not None else 0
        end_val = end if end is not None else sys.maxsize  # 2^63 – 1

        # If start_val and end_val are negative, add length to them
        if start_val < 0:
            start_val += length
        if end_val < 0:
            end_val += length

        # If the start val is still outside the tensor_size along the sliced
        # dimension, adjust it accordingly.
        if start_val < 0:
            start_val = 0
        elif start_val >= length:
            start_val = length

        # If the end val is still outside the tensor_size along the sliced
        # dimension, adjust it accordingly.
        if end_val < start_val:
            end_val = start_val
        elif end_val >= length:
            end_val = length

        # Return the adjusted start and end indices
        return (start_val, end_val)

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [
            exir_ops.edge.aten.slice_copy.Tensor,
            exir_ops.edge.aten.slice_scatter.default,
        ]

    def maybe_remove_or_replace(self, node: Node) -> bool:
        # Check if it is a slice_scatter op or not. The slice_scatter op has
        # an extra src argument at index 1.
        slice_scatter = node.target == exir_ops.edge.aten.slice_scatter.default

        # Get input tensor metadata
        input_node = node.args[0]
        if not isinstance(input_node, Node) or "val" not in input_node.meta:
            return False

        in_tensor = input_node.meta["val"]

        # Extract the slicing dimension
        dim_idx = 1 + (1 if slice_scatter else 0)
        dim = node.args[dim_idx] if len(node.args) > dim_idx else 0
        if not isinstance(dim, int):
            return False

        # Make dim non-negative
        original_dim = dim
        dim = dim if dim >= 0 else dim + in_tensor.dim()
        length = in_tensor.size(dim)

        # Get the adjusted start and end indices
        start_idx = 2 + (1 if slice_scatter else 0)
        end_idx = 3 + (1 if slice_scatter else 0)
        step_idx = 4 + (1 if slice_scatter else 0)

        start_val = node.args[start_idx] if len(node.args) > start_idx else None
        end_val = node.args[end_idx] if len(node.args) > end_idx else None
        step = node.args[step_idx] if len(node.args) > step_idx else 1

        # Validate types
        if start_val is not None and not isinstance(start_val, int):
            return False
        if end_val is not None and not isinstance(end_val, int):
            return False
        if not isinstance(step, int):
            return False

        # Get the adjusted start and end indices
        original_start = start_val
        original_end = end_val
        (adjusted_start, adjusted_end) = self.adjust_slice_range(
            length, start_val, end_val, step
        )

        # Check if anything changed
        nothing_changed = (
            adjusted_start == original_start
            and adjusted_end == original_end
            and dim == original_dim
        )
        if nothing_changed:
            return False

        # Replace the node based on the adjusted range
        with node.graph.inserting_before(node):
            if adjusted_start >= adjusted_end and slice_scatter:
                # For slice_scatter with empty range, return the input
                node.replace_all_uses_with(input_node)
            elif adjusted_start >= adjusted_end:
                # For slice with empty range, create an empty tensor
                empty_shape = list(in_tensor.shape)
                empty_shape[dim] = 0
                new_node = node.graph.call_function(
                    exir_ops.edge.aten.full.default,
                    (tuple(empty_shape), 0),
                    {"dtype": in_tensor.dtype},
                )
                new_node.meta = node.meta.copy()
                node.replace_all_uses_with(new_node)
            else:
                # Create new args with simplified indices
                if slice_scatter:
                    new_args = (
                        node.args[0],  # input
                        node.args[1],  # src
                        dim,
                        adjusted_start,
                        adjusted_end,
                        step,
                    )
                else:
                    new_args = (
                        node.args[0],  # input
                        dim,
                        adjusted_start,
                        adjusted_end,
                        step,
                    )
                # Target is guaranteed to be a callable since it's from our targets list
                target_callable = node.target
                assert callable(target_callable), "Target must be callable"
                new_node = node.graph.call_function(target_callable, new_args, {})
                new_node.meta = node.meta.copy()
                node.replace_all_uses_with(new_node)

        node.graph.erase_node(node)
        return True


@register_cadence_pass(CadencePassAttribute(opt_level=0))
class BindOptionalArgsPass(ExportPass):
    """Bind all optional args and kwargs."""

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        """
        Bind all optional args and kwargs for EdgeOpOverload operations.
        Only reports modified=True if arguments were actually changed.
        """
        modified = False
        for module in filter(
            lambda m: isinstance(m, torch.fx.GraphModule), graph_module.modules()
        ):
            for node in cast(torch.fx.GraphModule, module).graph.nodes:
                if node.op != "call_function":
                    continue
                if not isinstance(node.target, EdgeOpOverload):
                    continue

                # Try to rebind the args/kwargs to populate optional arguments
                updated_args = rebind(node.target, tuple(node.args), dict(node.kwargs))
                if updated_args is None:
                    # No schema matched or no changes needed
                    continue

                new_args, new_kwargs = updated_args
                # Check if anything actually changed
                if new_args != node.args or new_kwargs != node.kwargs:
                    node.args = new_args
                    node.kwargs = new_kwargs
                    modified = True

        if modified:
            graph_module.recompile()
            return super().call(graph_module)

        return PassResult(graph_module, modified)


# This class encapsulates all the functions that simplify the op's args
class CadenceSimplifyOpsInGraph:
    passes = [
        SimplifySliceOpPass,
        BindOptionalArgsPass,
    ]
