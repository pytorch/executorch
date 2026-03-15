# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pass to propagate transposes through TOSA Rescale operations.

TOSA Rescale operations are elementwise (per-element scaling and zero-point
adjustment), meaning they are layout-invariant. This pass identifies patterns
where transposes surround a Rescale operation and propagates them through,
enabling subsequent passes to fuse or eliminate consecutive transposes.

Pattern targeted:
    T(perm1) → Rescale → ... → T(perm2)

After propagation:
    Rescale → T(perm1) → ... → T(perm2)

This allows FuseConsecutiveTransposesPass to then merge T(perm1) and T(perm2)
if they are now adjacent or can be composed.

This is particularly effective for TOSA graphs where:
- ToTosaMemoryFormatPass inserts NCHW↔NHWC transposes
- Rescale operations are inserted for quantization/dequantization
- The pattern Transpose → Rescale → Conv → Rescale → Transpose is common
"""

import logging
from typing import List, Sequence, Set, Tuple, Type

import executorch.backends.arm.tosa.dialect  # noqa: F401 - loads TOSA dialect
import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch._ops import OpOverload

logger = logging.getLogger(__name__)


# Transpose/permute targets we can propagate
_PERMUTE_TARGETS: Tuple[OpOverload, ...] = (
    exir_ops.edge.aten.permute.default,
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.backend.tosa.TRANSPOSE.default,
)

# TOSA Rescale operation - elementwise and layout-invariant
_RESCALE_TARGETS: Tuple[OpOverload, ...] = (exir_ops.backend.tosa.RESCALE.default,)

# Additional layout-invariant operations through which we can propagate transposes
_LAYOUT_INVARIANT_OPS: Set[OpOverload] = {
    # Elementwise unary ops
    exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.sigmoid.default,
    exir_ops.edge.aten.tanh.default,
    exir_ops.edge.aten.neg.default,
    exir_ops.edge.aten.abs.default,
    exir_ops.edge.aten.exp.default,
    exir_ops.edge.aten.log.default,
    exir_ops.edge.aten.sqrt.default,
    exir_ops.edge.aten.rsqrt.default,
    exir_ops.edge.aten.clamp.default,
    exir_ops.edge.aten.hardswish.default,
    exir_ops.edge.aten.hardsigmoid.default,
    exir_ops.edge.aten.leaky_relu.default,
    exir_ops.edge.aten.gelu.default,
    exir_ops.edge.aten.silu.default,
    exir_ops.edge.aten.reciprocal.default,
}


def _get_permutation(node: torch.fx.Node) -> List[int] | None:
    """Extract permutation from a transpose/permute node."""
    if node.op != "call_function" or node.target not in _PERMUTE_TARGETS:
        return None
    if len(node.args) < 2:
        return None
    dims = node.args[1]
    if isinstance(dims, (list, tuple)):
        return list(dims)
    return None


def _is_rescale(node: torch.fx.Node) -> bool:
    """Check if a node is a TOSA Rescale operation."""
    if node.op != "call_function":
        return False
    return node.target in _RESCALE_TARGETS


def _is_layout_invariant(node: torch.fx.Node) -> bool:
    """Check if a node is a layout-invariant operation."""
    if node.op != "call_function":
        return False
    return node.target in _LAYOUT_INVARIANT_OPS or node.target in _RESCALE_TARGETS


def _permute_shape(shape: List[int], perm: List[int]) -> List[int]:
    """Apply permutation to a shape."""
    return [shape[i] for i in perm]


def _inverse_permutation(perm: Sequence[int]) -> List[int]:
    """Compute the inverse of a permutation."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _get_single_user(node: torch.fx.Node) -> torch.fx.Node | None:
    """Get the single user of a node, or None if multiple or no users."""
    users = list(node.users.keys())
    if len(users) == 1:
        return users[0]
    return None


class PropagateTransposesThroughRescalePass(ArmPass):
    """Fuse T1 -> Rescale -> T2 patterns by removing Rescale from the middle.

    This pass looks for patterns like:
        x -> permute(perm1) -> rescale -> permute(perm2) -> y

    Where Rescale is a layout-invariant operation, and transforms them by
    composing the permutations. Since Rescale is elementwise (operates on
    each element independently), it doesn't care about memory layout.

    This is different from simple consecutive transpose fusion because
    the Rescale operation sits between the two transposes.

    Pattern:
        Before: T1(perm1) -> Rescale -> T2(perm2)
        After:  Rescale -> T_combined(compose(perm1, perm2))

    If compose(perm1, perm2) is identity, both transposes are eliminated:
        Before: T1(perm1) -> Rescale -> T2(inverse(perm1))
        After:  Rescale (transposes removed)
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False

        # Iterate until no more fusions can be made
        while True:
            fused_this_iteration = False

            for node in list(graph_module.graph.nodes):
                # Look for the pattern: T1 -> Rescale -> T2
                # Start by finding T2 (second transpose)
                perm2 = _get_permutation(node)
                if perm2 is None:
                    continue

                # Get the input to T2 (should be Rescale)
                rescale_node = node.args[0]
                if not isinstance(rescale_node, torch.fx.Node):
                    continue

                if not _is_rescale(rescale_node):
                    continue

                # Check if Rescale has only one user (T2)
                if len(rescale_node.users) != 1:
                    continue

                # Get the input to Rescale (should be T1)
                transpose1_node = rescale_node.args[0]
                if not isinstance(transpose1_node, torch.fx.Node):
                    continue

                perm1 = _get_permutation(transpose1_node)
                if perm1 is None:
                    continue

                # Check if T1 has only one user (Rescale)
                if len(transpose1_node.users) != 1:
                    continue

                # Check if permutations have same rank
                if len(perm1) != len(perm2):
                    continue

                # We have the pattern: T1(perm1) -> Rescale -> T2(perm2)
                # Compose the permutations
                composed = [perm1[i] for i in perm2]

                # Get the original input (input to T1)
                original_input = transpose1_node.args[0]

                if composed == list(range(len(composed))):
                    # Identity permutation - remove both transposes
                    logger.debug(
                        f"Removing T({perm1}) -> Rescale -> T({perm2}) pattern "
                        f"(composes to identity)"
                    )

                    # Rewire: Rescale now takes the original input
                    new_args = list(rescale_node.args)
                    new_args[0] = original_input
                    rescale_node.args = tuple(new_args)

                    # Rewire: users of T2 now use Rescale
                    node.replace_all_uses_with(rescale_node)

                else:
                    # Non-identity - replace with single transpose after Rescale
                    logger.debug(
                        f"Fusing T({perm1}) -> Rescale -> T({perm2}) "
                        f"=> Rescale -> T({composed})"
                    )

                    # Rewire: Rescale now takes the original input
                    new_args = list(rescale_node.args)
                    new_args[0] = original_input
                    rescale_node.args = tuple(new_args)

                    # Create new combined transpose after Rescale
                    with graph_module.graph.inserting_after(rescale_node):
                        new_transpose = graph_module.graph.call_function(
                            exir_ops.edge.aten.permute_copy.default,
                            args=(rescale_node, composed),
                        )
                        # Copy metadata from T2
                        new_transpose.meta = node.meta.copy()

                    # Rewire: users of T2 now use new_transpose
                    node.replace_all_uses_with(new_transpose)

                fused_this_iteration = True
                modified = True
                break  # Restart iteration after modification

            if not fused_this_iteration:
                break

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        return PassResult(graph_module, modified)
