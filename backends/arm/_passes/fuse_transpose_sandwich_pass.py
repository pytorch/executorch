# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pass to fuse transpose-op-transpose patterns where the middle op is layout-invariant.

This pass identifies patterns like:
    T(perm1) → LayoutInvariantOp → T(perm2)

And if perm1 and perm2 cancel out (their composition is identity), removes both
transposes. This is particularly effective for TOSA graphs where ToTosaMemoryFormatPass
inserts NCHW↔NHWC transposes at operation boundaries.

Layout-invariant operations include elementwise ops (add, mul, relu, sigmoid, etc.)
that don't care about data layout because they operate independently on each element.
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


# Transpose/permute targets we can fuse
_PERMUTE_TARGETS: Tuple[OpOverload, ...] = (
    exir_ops.edge.aten.permute.default,
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.backend.tosa.TRANSPOSE.default,
)


# Layout-invariant operations - these don't depend on data layout
# They operate element-wise and produce the same result regardless of memory format
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
    exir_ops.edge.aten.sin.default,
    exir_ops.edge.aten.cos.default,
    exir_ops.edge.aten.floor.default,
    exir_ops.edge.aten.ceil.default,
    exir_ops.edge.aten.round.default,
    exir_ops.edge.aten.clamp.default,
    exir_ops.edge.aten.clamp.Tensor,
    exir_ops.edge.aten.hardswish.default,
    exir_ops.edge.aten.hardsigmoid.default,
    exir_ops.edge.aten.leaky_relu.default,
    exir_ops.edge.aten.gelu.default,
    exir_ops.edge.aten.silu.default,
    exir_ops.edge.aten.reciprocal.default,
    # Elementwise binary ops (when both inputs have same layout)
    exir_ops.edge.aten.add.Tensor,
    exir_ops.edge.aten.sub.Tensor,
    exir_ops.edge.aten.mul.Tensor,
    exir_ops.edge.aten.div.Tensor,
    exir_ops.edge.aten.maximum.default,
    exir_ops.edge.aten.minimum.default,
}


def _compose_permutations(perm1: Sequence[int], perm2: Sequence[int]) -> List[int]:
    """Compose two permutations: result[i] = perm1[perm2[i]].

    Given two consecutive permutations, computes the equivalent single permutation.
    """
    return [perm1[i] for i in perm2]


def _is_identity_permutation(perm: Sequence[int]) -> bool:
    """Check if a permutation is the identity (no-op)."""
    return list(perm) == list(range(len(perm)))


def _inverse_permutation(perm: Sequence[int]) -> List[int]:
    """Compute the inverse of a permutation."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


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


def _is_layout_invariant(node: torch.fx.Node) -> bool:
    """Check if a node is a layout-invariant operation."""
    if node.op != "call_function":
        return False
    return node.target in _LAYOUT_INVARIANT_OPS


def _get_single_non_constant_input(
    node: torch.fx.Node,
) -> torch.fx.Node | None:
    """Get the single non-constant input to a node, or None if multiple/none."""
    non_const_inputs = []
    for arg in node.args:
        if isinstance(arg, torch.fx.Node):
            # Check if it's a constant (placeholder that's a param or buffer)
            if arg.op == "get_attr":
                continue
            non_const_inputs.append(arg)
    if len(non_const_inputs) == 1:
        return non_const_inputs[0]
    return None


class FuseTransposeSandwichPass(ArmPass):
    """Fuse transpose-op-transpose patterns where permutations cancel out.

    This pass looks for patterns like:
        x -> permute(perm1) -> layout_invariant_op -> permute(perm2) -> y

    And transforms them to:
        x -> layout_invariant_op -> y  (if perm1 and perm2 cancel out)

    This is effective for removing TOSA transposes inserted by ToTosaMemoryFormatPass
    when they surround layout-invariant operations like ReLU, Add, etc.

    This pattern is common in TOSA graphs where:
    - Input transpose: NCHW → NHWC for TOSA
    - Layout-invariant op (relu, add, etc.)
    - Output transpose: NHWC → NCHW back to PyTorch format

    Example:
        Before: T([0,2,3,1]) -> ReLU -> T([0,3,1,2])
        After:  ReLU  (transposes removed, compose([0,2,3,1], [0,3,1,2]) = identity)
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False

        # Iterate until no more fusions can be made
        while True:
            fused_this_iteration = False

            for node in list(graph_module.graph.nodes):
                # Look for second transpose in pattern: T1 -> Op -> T2
                perm2 = _get_permutation(node)
                if perm2 is None:
                    continue

                # Get the middle operation (input to T2)
                middle_node = node.args[0]
                if not isinstance(middle_node, torch.fx.Node):
                    continue

                # Check if middle is layout-invariant
                if not _is_layout_invariant(middle_node):
                    continue

                # Check if middle has only one user (T2)
                if len(middle_node.users) != 1:
                    continue

                # Get the input to the middle op that's a transpose
                input_to_middle = _get_single_non_constant_input(middle_node)
                if input_to_middle is None:
                    continue

                perm1 = _get_permutation(input_to_middle)
                if perm1 is None:
                    continue

                # Check if T1 has only one user (the middle op)
                if len(input_to_middle.users) != 1:
                    continue

                # Check if permutations have same rank
                if len(perm1) != len(perm2):
                    continue

                # Compose permutations and check if they cancel
                composed = _compose_permutations(perm1, perm2)

                if _is_identity_permutation(composed):
                    # Permutations cancel out - remove both transposes
                    logger.debug(
                        f"Removing sandwich pattern: "
                        f"T({perm1}) -> {middle_node.target} -> T({perm2})"
                    )

                    # Get original input (input to T1)
                    original_input = input_to_middle.args[0]

                    # Rewire middle op to use original input
                    new_args = list(middle_node.args)
                    for i, arg in enumerate(new_args):
                        if arg is input_to_middle:
                            new_args[i] = original_input
                    middle_node.args = tuple(new_args)

                    # Rewire users of T2 to use middle op directly
                    node.replace_all_uses_with(middle_node)

                    fused_this_iteration = True
                    modified = True
                    break  # Restart iteration after modification

            if not fused_this_iteration:
                break

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        return PassResult(graph_module, modified)
