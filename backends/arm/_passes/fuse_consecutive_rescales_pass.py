# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import cast, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node

logger: logging.Logger = logging.getLogger(__name__)

# TOSA RESCALE argument positions:
#   args[0] = input tensor (Node)
#   args[1] = output dtype (e.g., torch.int8, torch.int32)
#   args[2] = scale list (List[float]; per-tensor when len == 1)
#   args[3] = input zero point (int)
#   args[4] = output zero point (int)
_ARG_INPUT = 0
_ARG_OUTPUT_DTYPE = 1
_ARG_SCALE = 2
_ARG_INPUT_ZP = 3
_ARG_OUTPUT_ZP = 4


class FuseConsecutiveRescalesPass(ArmPass):
    """Fuse consecutive RESCALE(INT32->INT8/INT16) -> RESCALE(INT8/INT16->INT32)
    pairs.

    InsertRescaleInt32Pass wraps each quantized arithmetic and comparison
    operator (add, sub, mul, abs, eq, ge, gt, le, lt, max, min, sum) with
    input rescales (INT8/INT16->INT32) and an output rescale
    (INT32->INT8/INT16). When two such ops are chained (e.g., add1 -> add2),
    the output rescale of add1 feeds directly into an input rescale of add2,
    creating a redundant INT32->INT8/INT16->INT32 round-trip that loses
    precision.

    This pass detects such pairs and handles two cases:

    - **Identity** (composed scale ~1.0, matching zero points): Removes both
      RESCALEs and directly wires R1's input to R2's users.  This eliminates
      the entire round-trip.  Bypassing the intermediate INT8/INT16 clamp can
      in theory cause up to ~120 INT8 steps of output difference when all
      inputs are near the clamp boundary; in practice, observed differences
      are 0-1 steps for typical distributions.  Tests use qtol=1.

    - **Non-identity**: Leaves the pair unchanged.  The Vela NPU compiler
      cannot correctly process INT32->INT32 RESCALE (produces all-zero NPU
      outputs), so non-identity pairs retain their INT8/INT16 intermediate.

    Handles multi-user R1 nodes: when R1 feeds both RESCALE and
    non-RESCALE users, each R1->R2 RESCALE pair is fused individually
    while preserving R1 for its non-RESCALE users.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        rescale_before = sum(1 for n in graph.nodes if _is_rescale(n))
        identity_pairs_fused = 0

        for node in list(graph.nodes):
            node = cast(Node, node)
            if not _is_fuseable_r1(node):
                continue

            r1_input = node.args[_ARG_INPUT]
            r1_input_zp = node.args[_ARG_INPUT_ZP]
            r1_scale = float(node.args[_ARG_SCALE][0])  # type: ignore[arg-type]

            node_fused = False
            for user in list(node.users):
                if _try_fuse_identity_pair(node, user, r1_input, r1_input_zp, r1_scale):
                    node_fused = True
                    identity_pairs_fused += 1

            if node_fused:
                modified = True

        if modified:
            graph.eliminate_dead_code()
            rescale_after = sum(1 for n in graph.nodes if _is_rescale(n))
            removed = rescale_before - rescale_after
            logger.info(
                "FuseConsecutiveRescalesPass: removed %d identity pairs "
                "(%d RESCALEs: %d -> %d)",
                identity_pairs_fused,
                removed,
                rescale_before,
                rescale_after,
            )
            graph_module.recompile()
            graph.lint()
            # Note: we deliberately skip super().call() — retracing is
            # unnecessary since this pass only rewires edges and removes
            # nodes without introducing new operations.

        return PassResult(graph_module, modified)


def _is_rescale(node: Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == exir_ops.backend.tosa.RESCALE.default
    )


def _is_fuseable_r1(node: Node) -> bool:
    """Check if node is an R1 candidate.

    R1 is RESCALE(INT32 -> INT8/INT16) with per-tensor scale.

    """
    if not _is_rescale(node):
        return False
    if node.args[_ARG_OUTPUT_DTYPE] not in (torch.int8, torch.int16):
        return False
    if len(node.args[_ARG_SCALE]) != 1:  # type: ignore[arg-type]
        return False
    r1_input = node.args[_ARG_INPUT]
    if not isinstance(r1_input, Node) or "val" not in r1_input.meta:
        return False
    if r1_input.meta["val"].dtype != torch.int32:
        return False
    return True


def _try_fuse_identity_pair(
    r1: Node,
    r2: Node,
    r1_input: Node,
    r1_input_zp: int,
    r1_scale: float,
) -> bool:
    """Try to fuse an R1->R2 identity pair.

    Returns True if fused.

    """
    if not _is_rescale(r2):
        return False
    if r2.args[_ARG_OUTPUT_DTYPE] != torch.int32:
        return False
    if r1.args[_ARG_OUTPUT_ZP] != r2.args[_ARG_INPUT_ZP]:
        return False
    if len(r2.args[_ARG_SCALE]) != 1:  # type: ignore[arg-type]
        return False

    r2_scale = float(r2.args[_ARG_SCALE][0])  # type: ignore[arg-type, index]
    composed_scale = r1_scale * r2_scale
    r2_output_zp = r2.args[_ARG_OUTPUT_ZP]

    if abs(composed_scale - 1.0) < 1e-6 and r1_input_zp == r2_output_zp:
        # Identity case: remove both RESCALEs and directly wire
        # R1's input (INT32) to R2's users.  The composed scale
        # is ~1.0 so the round-trip is a no-op modulo the INT8
        # clamp.  Bypassing the clamp can in theory cause up to
        # ~120 INT8 steps of difference near clamp boundaries;
        # observed differences are 0-1 steps.  Tests use qtol=1.
        r2.replace_all_uses_with(r1_input)
        return True

    # Non-identity: leave the pair unchanged.  Creating a
    # single INT32->INT32 RESCALE with the composed scale would
    # be semantically correct (and the TOSA ref model handles
    # it), but the Vela NPU compiler produces all-zero outputs
    # for INT32->INT32 RESCALE operations.
    return False
