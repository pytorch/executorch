# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Set, Type

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.to_tosa_memory_format_pass import (
    ToTosaMemoryFormatPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node

logger: logging.Logger = logging.getLogger(__name__)

_TRANSPOSE_TARGETS = (
    exir_ops.backend.tosa.TRANSPOSE.default,
    exir_ops.edge.aten.permute_copy.default,
    exir_ops.edge.aten.permute.default,
)


def _get_perm(node: Node) -> list[int] | None:
    """Extract the permutation list from a transpose/permute node."""
    if node.target not in _TRANSPOSE_TARGETS:
        return None
    perm = node.args[1] if len(node.args) > 1 else None
    if isinstance(perm, (list, tuple)):
        return list(perm)
    return None


class FuseConsecutiveTosaTransposesPass(ArmPass):
    """Fuse consecutive transpose/permute nodes that compose to identity.

    After ToTosaMemoryFormatPass inserts tosa.TRANSPOSE nodes for NCHW<->NHWC
    conversion, redundant consecutive pairs can appear between tosa.TRANSPOSE
    and aten.permute_copy nodes.

    When two adjacent nodes have complementary channels-last permutations
    (one is cl_order and the other is cl_inv), they compose to the identity
    permutation and both are erased.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        pairs_fused = 0

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue

            perm = _get_perm(node)
            if perm is None:
                continue

            rank = len(perm)
            sr = node.meta.get("tosa_spatial_rank", 0)
            if rank < 3 or sr < 1:
                continue

            cl_order = list(ToTosaMemoryFormatPass._channels_last_order(rank, sr))
            cl_inv = list(
                ToTosaMemoryFormatPass._channels_last_inverse_order(rank, sr)
            )
            if perm != cl_order and perm != cl_inv:
                continue

            complement = cl_inv if perm == cl_order else cl_order

            if len(node.users) != 1:
                continue
            user_node = next(iter(node.users))
            if not isinstance(user_node, Node):
                continue

            user_perm = _get_perm(user_node)
            if user_perm != complement:
                continue

            input_node = node.args[0]
            assert isinstance(input_node, Node)
            user_node.replace_all_uses_with(input_node)
            graph.erase_node(user_node)
            graph.erase_node(node)
            modified = True
            pairs_fused += 1

        if modified:
            logger.info(
                "FuseConsecutiveTosaTransposesPass: fused %d canceling pairs",
                pairs_fused,
            )
            graph_module.recompile()
            graph.lint()

        return PassResult(graph_module, modified)
