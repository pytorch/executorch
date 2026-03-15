# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fuse transpose -> reshape -> transpose patterns.

When reordering the graph (e.g from NCHW -> NHWC), a reshape is considered
non-reorderable, and transposes are added before the input and after the output.
In certain situations, we can transform this (transpose -> reshape -> transpose)
pattern into a single transpose followed by a (different) reshape.

Example: Consider a reshape on an NCHW tensor that reshapes the batch and channel
dimensions into the channel dimension:
    (N, C, H, W) -> reshape -> (1, (N, C), H, W)

If both input and output tensors are reordered to NHWC:
    (N, H, W, C)
    -> transpose -> (N, C, H, W)
    -> reshape -> (1, (N, C), H, W)
    -> transpose -> (1, H, W, (N, C))

This is equivalent to:
    (N, H, W, C) -> transpose -> (H, W, N, C) -> reshape -> (1, H, W, (N, C))

Inspired by bolt/nn/espresso/transforms/fuse_ops.py:fuse_transpose_reshape_transpose
"""

import logging
from typing import Optional, Set, Type

import executorch.backends.arm.tosa.dialect  # noqa: F401 - loads TOSA dialect
import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch import fx

logger = logging.getLogger(__name__)


# Supported permute/transpose targets
_PERMUTE_TARGETS = (
    torch.ops.aten.permute_copy.default,
    exir_ops.backend.tosa.TRANSPOSE.default,
)

# Supported reshape/view targets (both ATen and Edge dialects)
_RESHAPE_TARGETS = (
    torch.ops.aten.view_copy.default,
    torch.ops.aten._unsafe_view.default,
    exir_ops.edge.aten.view_copy.default,
)


def _get_permute_dims(node: fx.Node) -> Optional[list[int]]:
    """Extract the permutation dimensions from a permute node."""
    if node.target not in _PERMUTE_TARGETS:
        return None
    dims = node.args[1]
    if isinstance(dims, (list, tuple)):
        return list(dims)
    return None


def _get_reshape_shape(node: fx.Node) -> Optional[list[int]]:
    """Extract the shape from a view/reshape node."""
    if node.target not in _RESHAPE_TARGETS:
        return None
    shape = node.args[1]
    if isinstance(shape, (list, tuple)):
        return list(shape)
    return None


def _get_shape_indices(
    input_shape: list[int], output_shape: list[int]
) -> Optional[list[list[int]]]:
    """
    Compute which input dimensions map to each output dimension.

    For each output dimension, returns a list of input dimension indices
    that were combined to create it.

    Returns None if the reshape is not a simple combination/split of dimensions.
    """
    if not input_shape or not output_shape:
        return None

    input_idx = 0
    result = []

    for out_dim in output_shape:
        if out_dim == -1:
            return None

        indices = []
        accumulated = 1

        while accumulated < out_dim and input_idx < len(input_shape):
            indices.append(input_idx)
            accumulated *= input_shape[input_idx]
            input_idx += 1

        if accumulated == out_dim:
            if not indices and input_idx < len(input_shape):
                if input_shape[input_idx] == 1:
                    indices.append(input_idx)
                    input_idx += 1
                elif input_shape[input_idx] == out_dim:
                    indices.append(input_idx)
                    input_idx += 1
                else:
                    return None
            result.append(indices)
        elif accumulated > out_dim:
            return None
        else:
            return None

    if input_idx != len(input_shape):
        return None

    return result


class FuseTransposeReshapeTransposePass(ArmPass):
    """
    Fuses transpose -> reshape -> transpose patterns.

    This pass identifies patterns where:
    1. A permute is followed by a reshape/view
    2. Which is followed by another permute

    And transforms them into a single permute followed by a reshape with
    the combined effect.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self) -> None:
        super().__init__()
        self._graph_module: Optional[fx.GraphModule] = None

    def _find_patterns(
        self, graph_module: fx.GraphModule
    ) -> list[tuple[fx.Node, fx.Node, fx.Node]]:
        """Find all transpose -> reshape -> transpose patterns."""
        patterns = []
        graph = graph_module.graph

        for node in graph.nodes:
            if node.op != "call_function":
                continue

            dims1 = _get_permute_dims(node)
            if dims1 is None:
                continue

            transpose1 = node

            users = list(transpose1.users.keys())
            if len(users) != 1:
                continue

            reshape = users[0]
            if reshape.op != "call_function":
                continue

            reshape_shape = _get_reshape_shape(reshape)
            if reshape_shape is None:
                continue

            reshape_users = list(reshape.users.keys())
            if len(reshape_users) != 1:
                continue

            transpose2 = reshape_users[0]
            if transpose2.op != "call_function":
                continue

            dims2 = _get_permute_dims(transpose2)
            if dims2 is None:
                continue

            patterns.append((transpose1, reshape, transpose2))

        return patterns

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        self._graph_module = graph_module
        modified = False

        patterns = self._find_patterns(graph_module)

        for transpose1, reshape, transpose2 in patterns:
            dims1 = _get_permute_dims(transpose1)
            dims2 = _get_permute_dims(transpose2)
            reshape_shape = _get_reshape_shape(reshape)

            if dims1 is None or dims2 is None or reshape_shape is None:
                continue

            input_node = transpose1.args[0]
            if not isinstance(input_node, fx.Node):
                continue

            input_val = input_node.meta.get("val")
            if input_val is None:
                continue

            input_shape = list(input_val.shape)

            transposed1_shape = [input_shape[i] for i in dims1]

            shape_indices = _get_shape_indices(transposed1_shape, reshape_shape)
            if shape_indices is None:
                logger.warning(
                    f"FuseTransposeReshapeTransposePass: Cannot compute shape indices for "
                    f"reshape {reshape.name}: transposed1_shape={transposed1_shape}, "
                    f"reshape_shape={reshape_shape}"
                )
                continue

            original = list(range(len(dims1)))
            transposed1 = [original[i] for i in dims1]

            reshaped = [tuple(transposed1[i] for i in s) for s in shape_indices]

            transposed2 = [reshaped[i] for i in dims2]

            new_transpose_axes = [i for s in transposed2 for i in s]

            if len(new_transpose_axes) != len(input_shape):
                logger.debug(
                    "New transpose axes length mismatch, skipping fusion"
                )
                continue

            output_val = transpose2.meta.get("val")
            if output_val is None:
                continue
            output_shape = list(output_val.shape)

            logger.info(
                f"Fusing transpose({dims1}) -> reshape({reshape_shape}) -> transpose({dims2}) "
                f"into transpose({new_transpose_axes}) -> reshape({output_shape})"
            )

            with graph_module.graph.inserting_before(transpose1):
                new_permute = graph_module.graph.call_function(
                    torch.ops.aten.permute_copy.default,
                    (input_node, new_transpose_axes),
                )

                if input_val is not None:
                    new_permute.meta["val"] = input_val.permute(new_transpose_axes)

                new_reshape = graph_module.graph.call_function(
                    torch.ops.aten.view_copy.default,
                    (new_permute, output_shape),
                )

                if output_val is not None:
                    new_reshape.meta["val"] = output_val

            transpose2.replace_all_uses_with(new_reshape)

            graph_module.graph.erase_node(transpose2)
            graph_module.graph.erase_node(reshape)
            graph_module.graph.erase_node(transpose1)

            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
