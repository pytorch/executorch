# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pass to propagate transposes through Concat operations.

Concat is a layout-aware operation that concatenates tensors along a specific
dimension. When all inputs to a Concat have the same transpose permutation,
and the output is transposed with the inverse permutation, the transposes
can be eliminated by adjusting the concat dimension.

Pattern targeted:
    [T(perm1), T(perm1), ...] → Concat(dim=d) → T(perm2)

If perm1 and perm2 compose to identity (i.e., perm2 is the inverse of perm1),
the pattern can be simplified to:
    [inputs...] → Concat(dim=perm1[d]) → outputs

This is particularly effective for TOSA graphs where:
- ToTosaMemoryFormatPass inserts NCHW↔NHWC transposes at graph boundaries
- Concat operations often have transposed inputs that are re-transposed after
- Example: T(NCHW→NHWC) → Concat → T(NHWC→NCHW)
"""

import logging
from typing import List, Set, Tuple, Type

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

# Concat targets
_CONCAT_TARGETS: Tuple[OpOverload, ...] = (
    exir_ops.edge.aten.cat.default,
)


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


def _is_concat(node: torch.fx.Node) -> bool:
    """Check if a node is a Concat operation."""
    if node.op != "call_function":
        return False
    return node.target in _CONCAT_TARGETS


def _compose_permutations(perm1: List[int], perm2: List[int]) -> List[int]:
    """Compose two permutations: result[i] = perm1[perm2[i]]."""
    return [perm1[i] for i in perm2]


def _is_identity_permutation(perm: List[int]) -> bool:
    """Check if a permutation is identity."""
    return perm == list(range(len(perm)))


def _inverse_permutation(perm: List[int]) -> List[int]:
    """Compute the inverse of a permutation."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


class PropagateTransposesThroughConcatPass(ArmPass):
    """Propagate transposes through Concat operations.

    This pass looks for patterns like:
        [T(perm), T(perm), ...] → Concat(dim=d) → T(inv_perm)

    Where all inputs to Concat have the same permutation perm, and the output
    is transposed with the inverse permutation. In this case, the transposes
    cancel out, and we can eliminate them by adjusting the concat dimension.

    Transformation:
        Before: [T(perm)(x1), T(perm)(x2)] → Concat(dim=d) → T(inv_perm) → y
        After:  [x1, x2] → Concat(dim=perm[d]) → y

    The concat dimension is adjusted because:
    - Original: Concat on transposed data along dim d
    - New: Concat on original data along the dimension that maps to d after transpose

    Example for NCHW→NHWC (perm=[0,2,3,1]):
    - Original: Concat(dim=3) on NHWC data → Transpose to NCHW
    - After: Concat(dim=1) on NCHW data (since perm[3]=1, but we need inverse)
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False

        # Iterate until no more fusions can be made
        while True:
            fused_this_iteration = False

            for node in list(graph_module.graph.nodes):
                # Look for pattern: Concat → T (start from the output transpose)
                output_perm = _get_permutation(node)
                if output_perm is None:
                    continue

                # Get the input to the output transpose (should be Concat)
                concat_node = node.args[0]
                if not isinstance(concat_node, torch.fx.Node):
                    continue

                if not _is_concat(concat_node):
                    continue

                # Check if Concat has only one user (the output transpose)
                if len(concat_node.users) != 1:
                    continue

                # Get the concat inputs and dimension
                concat_inputs = concat_node.args[0]
                if not isinstance(concat_inputs, (list, tuple)):
                    continue

                # Get concat dimension (default is 0)
                concat_dim = concat_node.args[1] if len(concat_node.args) > 1 else 0

                # Check if all concat inputs are transposes with the same permutation
                input_perms = []
                input_sources = []
                all_valid = True

                for inp in concat_inputs:
                    if not isinstance(inp, torch.fx.Node):
                        all_valid = False
                        break

                    inp_perm = _get_permutation(inp)
                    if inp_perm is None:
                        all_valid = False
                        break

                    # Check single user (only this Concat)
                    if len(inp.users) != 1:
                        all_valid = False
                        break

                    input_perms.append(inp_perm)
                    input_sources.append(inp.args[0])

                if not all_valid:
                    continue

                if len(input_perms) == 0:
                    continue

                # Check all input permutations are the same
                first_perm = input_perms[0]
                if not all(perm == first_perm for perm in input_perms):
                    continue

                # Check if input and output permutations have the same rank
                if len(first_perm) != len(output_perm):
                    continue

                # Check if input_perm → output_perm composes to identity
                composed = _compose_permutations(first_perm, output_perm)

                if not _is_identity_permutation(composed):
                    # Permutations don't cancel out
                    continue

                # We have the pattern! All transposes can be eliminated.
                # New concat dimension: when we remove the input transposes,
                # the data is in the original layout. The concat dimension
                # needs to be adjusted.
                #
                # Original: data → T(perm) → Concat(dim=d) → T(inv_perm)
                # The concat happens on transposed data at dimension d.
                # In original layout, this corresponds to dimension inv_perm[d].
                #
                # But wait - we're removing BOTH the input and output transposes.
                # So the new concat dimension should be: first_perm.index(concat_dim)
                # i.e., find where concat_dim came from in the original layout.
                #
                # Actually, the inverse permutation tells us this:
                # inv_perm[d] gives the original dimension that maps to d.

                inv_first_perm = _inverse_permutation(first_perm)
                new_concat_dim = inv_first_perm[concat_dim]

                logger.debug(
                    f"Propagating transposes through Concat: "
                    f"T({first_perm}) x {len(input_perms)} inputs → Concat(dim={concat_dim}) → T({output_perm}) "
                    f"=> Concat(dim={new_concat_dim})"
                )

                # Create new concat node with adjusted dimension
                with graph_module.graph.inserting_before(concat_node):
                    new_concat = graph_module.graph.call_function(
                        concat_node.target,
                        args=(list(input_sources), new_concat_dim),
                        kwargs=dict(concat_node.kwargs),
                    )
                    new_concat.meta = node.meta.copy()

                # Replace the output transpose with the new concat
                node.replace_all_uses_with(new_concat)

                fused_this_iteration = True
                modified = True
                break  # Restart iteration after modification

            if not fused_this_iteration:
                break

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        return PassResult(graph_module, modified)
