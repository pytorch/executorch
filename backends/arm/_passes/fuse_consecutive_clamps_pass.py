# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import cast, Set, Type

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    QuantizeClampArgumentsPass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node

logger: logging.Logger = logging.getLogger(__name__)

# aten.clamp.default argument positions: (input, min, max). min and/or max may
# be None (meaning -inf / +inf respectively).
_ARG_INPUT = 0
_ARG_MIN = 1
_ARG_MAX = 2

_CLAMP_OP = exir_ops.edge.aten.clamp.default


class FuseConsecutiveClampsPass(ArmPass):
    """Fuse a chain of consecutive ``clamp.default`` ops into a single clamp.

    ``ConvertToClampPass`` normalizes ``hardtanh``, ``relu`` (and relu6 via its
    hardtanh decomposition) into ``aten.clamp.default``. As a result, common
    activation chains such as ``HardTanh(a, b) -> ReLU`` become two adjacent
    clamps: ``clamp(a, b) -> clamp(0, None)``. The second clamp is redundant.

    Clamp composition is exact::

        clamp(clamp(x, a, b), c, d) == clamp(x, max(a, c), min(b, d))

    (treating ``None`` bounds as -inf / +inf). This pass rewrites every
    ``clamp -> clamp`` pair where the first clamp feeds *only* the second into a
    single clamp with the composed bounds, then removes the now-dead first
    clamp. Chains of length >2 collapse to one clamp by iterating to a fixed
    point.

    Quantized path: when run after ``FoldAndAnnotateQParamsPass`` the clamps
    carry ``input_qparams`` / ``output_qparams`` in their meta. The surviving
    clamp takes the first clamp's input qparams and keeps the second clamp's
    output qparams, dropping the intermediate requantization. Bypassing that
    requant can differ by at most ~1 quantization step at the clamp boundary
    (same tradeoff as ``FuseConsecutiveRescalesPass``; tests use qtol=1).

    """

    # We emit clamp.default with float min/max args; QuantizeClampArgumentsPass
    # must still run afterwards to quantize those args (same requirement as
    # ConvertToClampPass). DecomposeTOSAUnsupportedClampPass runs earlier in the
    # pipeline and only decomposes int32 clamps, which our fused clamp inherits
    # from its (already-processed) inputs, so it is not required after us.
    _passes_required_after: Set[Type[ExportPass]] = {QuantizeClampArgumentsPass}

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        clamp_before = sum(1 for n in graph.nodes if _is_clamp(n))
        fused = 0

        # graph.nodes is topologically ordered, so a single forward sweep
        # collapses an entire chain (a -> b -> c): after folding a into b, b is
        # still visited later in the same sweep and folds into c. The outer loop
        # only re-runs to catch a pair newly exposed by a prior fusion.
        while True:
            fused_this_pass = 0
            for node in list(graph.nodes):
                node = cast(Node, node)
                if not _is_clamp(node):
                    continue
                if len(node.users) != 1:
                    continue
                user = next(iter(node.users))
                if not _is_clamp(user):
                    continue
                if _get_input(user) is not node:
                    continue
                if _fuse_pair(node, user):
                    graph.erase_node(node)
                    fused_this_pass += 1
            fused += fused_this_pass
            if fused_this_pass == 0:
                break

        if fused:
            graph.eliminate_dead_code()
            clamp_after = sum(1 for n in graph.nodes if _is_clamp(n))
            logger.info(
                "FuseConsecutiveClampsPass: fused %d clamp pairs (%d -> %d clamps)",
                fused,
                clamp_before,
                clamp_after,
            )
            graph_module.recompile()
            graph.lint()
            # Deliberately skip super().call(): this pass only rewires edges,
            # edits scalar args, and removes nodes -- no new ops to retrace.

        return PassResult(graph_module, fused > 0)


def _is_clamp(node: Node) -> bool:
    return node.op == "call_function" and node.target == _CLAMP_OP


def _get_input(node: Node) -> Node | None:
    if len(node.args) > _ARG_INPUT:
        arg = node.args[_ARG_INPUT]
    else:
        arg = node.kwargs.get("self")
    return arg if isinstance(arg, Node) else None


def _get_bounds(node: Node) -> tuple[float | None, float | None]:
    # Bounds may be spelled positionally or as min=/max= kwargs; fall back to
    # kwargs so an explicitly-authored clamp doesn't have a bound silently
    # dropped (which would widen the fused range).
    args = node.args
    kwargs = node.kwargs
    min_val = args[_ARG_MIN] if len(args) > _ARG_MIN else kwargs.get("min")
    max_val = args[_ARG_MAX] if len(args) > _ARG_MAX else kwargs.get("max")
    return cast("float | None", min_val), cast("float | None", max_val)


def _compose_lower(a: float | None, b: float | None) -> float | None:
    """Compose two clamp lower bounds -- the tighter (larger) wins."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _compose_upper(a: float | None, b: float | None) -> float | None:
    """Compose two clamp upper bounds -- the tighter (smaller) wins."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _fuse_pair(first: Node, second: Node) -> bool:
    """Fold ``first`` into ``second`` in place. Returns True if fused.

    ``second`` keeps its identity (downstream users point at it) and its output
    qparams; it is rewired to read ``first``'s input with the composed bounds.

    """
    first_input = _get_input(first)
    if first_input is None:
        return False
    min1, max1 = _get_bounds(first)
    min2, max2 = _get_bounds(second)
    new_min = _compose_lower(min1, min2)
    new_max = _compose_upper(max1, max2)

    # Empty range (min > max) is pathological -- torch.clamp with min>max is not
    # associative in that regime, so leave the pair untouched.
    if new_min is not None and new_max is not None and new_min > new_max:
        return False

    # Rewire the survivor to read first's input, writing a canonical
    # (input, min, max) layout and dropping any stale min/max spelled as kwargs
    # on the original clamp.
    second.args = (first_input, new_min, new_max)
    if second.kwargs:
        second.kwargs = {
            k: v for k, v in second.kwargs.items() if k not in ("min", "max")
        }

    # Transfer input quantization params from the first clamp so the surviving
    # clamp is consistent with its new input (quantized path only).
    first_in_qparams = first.meta.get("input_qparams")
    if first_in_qparams and "input_qparams" in second.meta:
        second.meta["input_qparams"] = dict(first_in_qparams)

    return True
