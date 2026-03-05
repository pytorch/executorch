# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fuse transpose -> reshape -> linear patterns.

A common artifact from reordering dimensions is that transposes are
inserted before FC/Linear layers. Usually this looks something like:
   Transpose -> Reshape -> Linear
where the Reshape flattens all dimensions except the batch dimension.

This pass eliminates the transpose by applying the inverse of the transpose
to the Linear layer's weights instead.

Inspired by bolt/nn/espresso/transforms/fuse_ops.py:fuse_transpose_reshape_fc
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
    if node.target not in (
        torch.ops.aten.view_copy.default,
        torch.ops.aten._unsafe_view.default,
    ):
        return None
    shape = node.args[1]
    if isinstance(shape, (list, tuple)):
        return list(shape)
    return None


def _is_linear_node(node: fx.Node) -> bool:
    """Check if a node is a linear operation."""
    if node.op != "call_function":
        return False
    return node.target in (
        torch.ops.aten.linear.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
    )


def _get_weight_node(node: fx.Node) -> Optional[fx.Node]:
    """Get the weight tensor node from a linear operation."""
    if node.target == torch.ops.aten.linear.default:
        if len(node.args) >= 2:
            return node.args[1]
    elif node.target == torch.ops.aten.mm.default:
        if len(node.args) >= 2:
            return node.args[1]
    elif node.target == torch.ops.aten.addmm.default:
        if len(node.args) >= 3:
            return node.args[2]
    return None


class FuseTransposeReshapeLinearPass(ArmPass):
    """
    Fuses transpose -> reshape -> linear patterns by folding the transpose
    into the linear layer's weights.

    This pass identifies patterns where:
    1. A permute is followed by a reshape
    2. Which is followed by a linear/mm operation
    3. The transpose does not modify the batch dimension
    4. The reshape flattens all non-batch dimensions

    Instead of transposing at runtime, the pass applies the inverse transpose
    to the linear weights at compile time.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self) -> None:
        super().__init__()
        self._graph_module: Optional[fx.GraphModule] = None

    def _find_patterns(
        self, graph_module: fx.GraphModule
    ) -> list[tuple[fx.Node, fx.Node, fx.Node]]:
        """Find all transpose -> reshape -> linear patterns."""
        patterns = []
        graph = graph_module.graph

        for node in graph.nodes:
            if node.op != "call_function":
                continue

            dims = _get_permute_dims(node)
            if dims is None:
                continue

            transpose = node

            users = list(transpose.users.keys())
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

            linear = reshape_users[0]
            if not _is_linear_node(linear):
                continue

            patterns.append((transpose, reshape, linear))

        return patterns

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        self._graph_module = graph_module
        modified = False

        patterns = self._find_patterns(graph_module)

        for transpose, reshape, linear in patterns:
            dims = _get_permute_dims(transpose)
            reshape_shape = _get_reshape_shape(reshape)

            if dims is None or reshape_shape is None:
                continue

            if dims[0] != 0:
                logger.debug(
                    "Transpose modifies batch dimension, skipping fusion"
                )
                continue

            if len(reshape_shape) != 2 or reshape_shape[0] not in (-1, 1):
                logger.debug(
                    "Reshape does not flatten to 2D with batch dim, skipping"
                )
                continue

            input_node = transpose.args[0]
            if not isinstance(input_node, fx.Node):
                continue

            input_val = input_node.meta.get("val")
            if input_val is None:
                continue
            input_shape = list(input_val.shape)

            transpose_val = transpose.meta.get("val")
            if transpose_val is None:
                continue

            if input_shape[0] != transpose_val.shape[0]:
                logger.debug("Batch dimension changed during transpose, skipping")
                continue

            reshape_val = reshape.meta.get("val")
            if reshape_val is None:
                continue

            if reshape_val.shape[0] != transpose_val.shape[0]:
                logger.debug("Batch dimension changed during reshape, skipping")
                continue

            weight_node = _get_weight_node(linear)
            if weight_node is None or not isinstance(weight_node, fx.Node):
                logger.debug("Cannot find weight node for linear, skipping")
                continue

            weight_val = weight_node.meta.get("val")
            if weight_val is None:
                continue

            inv_transpose = [dims.index(i) for i in range(len(dims))]

            inner_shape = transpose_val.shape[1:]

            try:
                new_weight_shape = (-1,) + tuple(inner_shape)
                weight_data = weight_val.reshape(new_weight_shape)
                weight_data = weight_data.permute(inv_transpose)
                weight_data = weight_data.reshape(weight_val.shape)
            except (RuntimeError, ValueError) as e:
                logger.debug(f"Cannot reshape weights: {e}, skipping")
                continue

            logger.info(
                f"Fusing transpose({dims}) into linear weights, "
                f"inverse permutation: {inv_transpose}"
            )

            with graph_module.graph.inserting_before(transpose):
                new_reshape = graph_module.graph.call_function(
                    torch.ops.aten.view_copy.default,
                    (input_node, reshape_shape),
                )
                new_reshape.meta["val"] = reshape_val

            reshape.replace_all_uses_with(new_reshape)

            graph_module.graph.erase_node(reshape)
            graph_module.graph.erase_node(transpose)

            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, modified)
