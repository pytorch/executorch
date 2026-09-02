# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Set, Type

from executorch.backends.arm._passes.arm_pass import ArmOpTargetedPass
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm._passes.decompose_softmax_pass import DecomposeSoftmaxPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node

logger = logging.getLogger(__name__)

_where_ops = (exir_ops.edge.aten.where.self,)
_softmax_ops = (exir_ops.edge.aten._softmax.default,)
_eq_ops = (exir_ops.edge.aten.eq.Scalar,)
_logical_not_ops = (exir_ops.edge.aten.logical_not.default,)
_any_dim_ops = (exir_ops.edge.aten.any.dim,)
_full_like_ops = (exir_ops.edge.aten.full_like.default,)


def _is_call_function_target(node: Node, targets: tuple[object, ...]) -> bool:
    """Return True when ``node`` calls one of ``targets``."""
    return node.op == "call_function" and node.target in targets


class RemoveSafeSoftmaxGuardPass(ArmOpTargetedPass):
    """Remove exact expanded SDPA safe-softmax guards.

    The matched pattern is:

        softmax = _softmax(scores, dim, ...)
        pred = logical_not(any(logical_not(eq(scores, -inf)), dim, True))
        zeros = full_like(softmax, 0)
        out = where(pred, zeros, softmax)

    Replacing ``out`` with ``softmax`` changes rows containing only ``-inf``
    scores from zeros to regular-softmax results, which can contain NaNs. Run
    this pass only when the configured policy permits that behavior.

    """

    _passes_required_after: Set[Type[ExportPass]] = {DecomposeSoftmaxPass}
    target_ops = _where_ops

    def call(self, graph_module: GraphModule) -> PassResult:
        removed = 0
        for node in graph_module.graph.nodes:
            softmax = self._match_safe_softmax_guard(node)
            if softmax is None:
                continue

            node.replace_all_uses_with(softmax)
            removed += 1

        if removed:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            logger.info("Removed %d expanded SDPA safe-softmax guard(s).", removed)

        return PassResult(graph_module, removed > 0)

    def _match_safe_softmax_guard(self, where: Node) -> Node | None:
        """Return the softmax node when ``where`` matches the guard pattern."""
        branches = self._match_where_branches(where)
        if branches is None:
            return None
        pred, zero_branch, softmax = branches

        predicate = self._match_guard_predicate(pred)
        if predicate is None:
            return None
        any_dim, eq = predicate

        scores, neg_inf = eq.args
        if not isinstance(scores, Node):
            return None
        if not (isinstance(neg_inf, float) and neg_inf == -math.inf):
            return None
        if softmax.args[0] is not scores:
            return None
        # The guard must reduce along the same dimension that softmax uses.
        # Otherwise the predicate describes a different row structure.
        if not self._same_dim(softmax, any_dim):
            return None

        return softmax

    def _match_where_branches(self, where: Node) -> tuple[Node, Node, Node] | None:
        """Match ``where(pred, zeros, softmax)`` guard output branches."""
        if not _is_call_function_target(where, _where_ops):
            return None
        pred, zero_branch, softmax = where.args
        if not (
            isinstance(pred, Node)
            and isinstance(zero_branch, Node)
            and isinstance(softmax, Node)
        ):
            return None

        if not (
            _is_call_function_target(softmax, _softmax_ops)
            and _is_call_function_target(zero_branch, _full_like_ops)
            and self._is_zero_full_like_of_softmax(zero_branch, softmax)
        ):
            return None
        if get_first_fake_tensor(where).dtype != get_first_fake_tensor(softmax).dtype:
            return None

        return pred, zero_branch, softmax

    @staticmethod
    def _match_guard_predicate(pred: Node) -> tuple[Node, Node] | None:
        """Match the predicate for rows containing only ``-inf`` values."""
        if not _is_call_function_target(pred, _logical_not_ops):
            return None

        any_dim = pred.args[0]
        if not (
            isinstance(any_dim, Node)
            and _is_call_function_target(any_dim, _any_dim_ops)
            and len(any_dim.args) > 2
            and any_dim.args[2] is True
        ):
            return None

        logical_not = any_dim.args[0]
        if not (
            isinstance(logical_not, Node)
            and _is_call_function_target(logical_not, _logical_not_ops)
        ):
            return None

        eq = logical_not.args[0]
        if not (isinstance(eq, Node) and _is_call_function_target(eq, _eq_ops)):
            return None

        return any_dim, eq

    @staticmethod
    def _is_zero_full_like_of_softmax(full_like: Node, softmax: Node) -> bool:
        """Return True when ``full_like`` matches ``full_like(softmax, 0)``."""
        zero_value = full_like.args[1]
        return (
            full_like.args[0] is softmax
            and isinstance(zero_value, (int, float))
            and zero_value == 0
        )

    @staticmethod
    def _same_dim(softmax: Node, any_dim: Node) -> bool:
        """Return True when softmax and guard reduction dims are equivalent.

        The exact values may differ when one dim is negative, for example
        ``-1`` and ``3`` for a rank-4 tensor. In that case, use the softmax
        input metadata to normalize both dims by rank.

        """
        softmax_dim = softmax.args[1]
        any_dim_arg = any_dim.args[1]
        if softmax_dim == any_dim_arg:
            return True
        # Dynamic dims are not safe to normalize by rank.
        if not isinstance(softmax_dim, int) or not isinstance(any_dim_arg, int):
            return False

        rank = get_first_fake_tensor(softmax.all_input_nodes[0]).dim()
        return rank > 0 and softmax_dim % rank == any_dim_arg % rank
