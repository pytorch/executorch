# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pass to fuse consecutive transpose/permute operations.

This pass identifies chains of transpose/permute operations and either:
1. Removes both if they cancel out (result is identity permutation)
2. Fuses them into a single permute with combined dimensions

This optimization reduces runtime overhead by eliminating redundant memory
movement operations.
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


_PERMUTE_TARGETS: Tuple[OpOverload, ...] = (
    exir_ops.edge.aten.permute.default,
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.backend.tosa.TRANSPOSE.default,
)


def _compose_permutations(perm1: Sequence[int], perm2: Sequence[int]) -> List[int]:
    """Compose two permutations: result[i] = perm1[perm2[i]].

    Given two consecutive permutations, computes the equivalent single permutation.

    Args:
        perm1: First permutation (applied first to data)
        perm2: Second permutation (applied second to data)

    Returns:
        Combined permutation that has the same effect as applying perm1 then perm2
    """
    return [perm1[i] for i in perm2]


def _is_identity_permutation(perm: Sequence[int]) -> bool:
    """Check if a permutation is the identity (no-op).

    Args:
        perm: Permutation to check

    Returns:
        True if perm[i] == i for all i (identity permutation)
    """
    return list(perm) == list(range(len(perm)))


class FuseConsecutiveTransposesPass(ArmPass):
    """Fuse consecutive transpose/permute operations.

    This pass looks for patterns like:
        x -> permute(dims1) -> permute(dims2) -> y

    And transforms them to either:
        x -> y  (if dims1 and dims2 cancel out)
    Or:
        x -> permute(combined_dims) -> y  (single fused permute)

    This is inspired by bolt/nn/espresso/transforms/fuse_ops.py:fuse_transposes
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False

        # Iterate until no more fusions can be made
        while True:
            fused_this_iteration = False

            for node in list(graph_module.graph.nodes):
                if node.op != "call_function":
                    continue
                if node.target not in _PERMUTE_TARGETS:
                    continue

                # Check if input is also a permute
                input_node = node.args[0]
                if not isinstance(input_node, torch.fx.Node):
                    continue
                if input_node.op != "call_function":
                    continue
                if input_node.target not in _PERMUTE_TARGETS:
                    continue

                # We have permute -> permute pattern
                permute1 = input_node
                permute2 = node

                # Get the permutation dimensions
                dims1 = permute1.args[1]
                dims2 = permute2.args[1]

                if not isinstance(dims1, (list, tuple)):
                    continue
                if not isinstance(dims2, (list, tuple)):
                    continue

                # Normalize to lists
                dims1 = list(dims1)
                dims2 = list(dims2)

                if len(dims1) != len(dims2):
                    # Permutations must have same rank
                    continue

                # Compose the permutations
                combined_dims = _compose_permutations(dims1, dims2)

                if _is_identity_permutation(combined_dims):
                    # Two permutes cancel out - remove both
                    logger.debug(
                        f"Removing canceling permutes: "
                        f"permute({dims1}) -> permute({dims2})"
                    )
                    permute2.replace_all_uses_with(permute1.args[0])
                else:
                    # Fuse into single permute
                    logger.debug(
                        f"Fusing permutes: "
                        f"permute({dims1}) -> permute({dims2}) => permute({combined_dims})"
                    )

                    with graph_module.graph.inserting_before(permute1):
                        new_permute = graph_module.graph.call_function(
                            exir_ops.edge.aten.permute_copy.default,
                            args=(permute1.args[0], combined_dims),
                        )
                        # Copy metadata from the output permute
                        new_permute.meta = permute2.meta.copy()
                        permute2.replace_all_uses_with(new_permute)

                fused_this_iteration = True
                modified = True
                break  # Restart iteration after modification

            if not fused_this_iteration:
                break

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()

        return PassResult(graph_module, modified)
