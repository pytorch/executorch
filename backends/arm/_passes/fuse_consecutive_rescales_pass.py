# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult


class FuseConsecutiveRescalesPass(ArmPass):
    """Fuse consecutive RESCALE(INT32->INT8/INT16) -> RESCALE(INT8/INT16->INT32)
    pairs.

    InsertRescaleInt32Pass wraps each add/mul/sub with input rescales
    (INT8/INT16->INT32) and an output rescale (INT32->INT8/INT16). When
    two such ops are chained (e.g., add1 -> add2), the output rescale
    of add1 feeds directly into an input rescale of add2, creating a
    redundant INT32->INT8/INT16->INT32 round-trip that loses precision.

    This pass detects such pairs and either:
    - Skips pairs where the composed scale is ~1.0 and zero points match
      (identity case), preserving the INT8/INT16 clamp semantics that the
      reference quantized model retains
    - Replaces non-identity pairs with a single INT32->INT32 RESCALE
      with the composed scale

    Handles multi-user R1 nodes: when R1 feeds both RESCALE and
    non-RESCALE users, each R1->R2 RESCALE pair is fused individually
    while preserving R1 for its non-RESCALE users.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        nodes_to_erase = []

        for node in list(graph.nodes):
            node = cast(Node, node)
            if not _is_rescale(node):
                continue

            # R1 = node: output rescale (INT32 -> INT8/INT16)
            r1_output_dtype = node.args[1]
            if r1_output_dtype not in (torch.int8, torch.int16):
                continue

            # Skip per-channel scales (multi-element scale list). Only fuse
            # per-tensor RESCALEs where a single composed scale is correct.
            if len(node.args[2]) != 1:
                continue

            r1_input = node.args[0]
            r1_input_zp = node.args[3]
            r1_output_zp = node.args[4]
            r1_scale = float(node.args[2][0])

            # Check each user individually (handles multi-user R1)
            for user in list(node.users):
                if not _is_rescale(user):
                    continue

                # R2 = user: input rescale (INT8/INT16 -> INT32)
                r2_output_dtype = user.args[1]
                if r2_output_dtype != torch.int32:
                    continue

                r2_input_zp = user.args[3]

                # Guard: intermediate zero points must match for correct
                # composition. Without this, the offset term
                # (r1_output_zp - r2_input_zp) * r2_scale is silently lost.
                if r1_output_zp != r2_input_zp:
                    continue

                if len(user.args[2]) != 1:
                    continue

                r2_scale = float(user.args[2][0])
                composed_scale = r1_scale * r2_scale
                r2_output_zp = user.args[4]

                if abs(composed_scale - 1.0) < 1e-6 and r1_input_zp == r2_output_zp:
                    # Identity case: skip fusion to preserve the intermediate
                    # INT8/INT16 clamp at [-128,127].  Removing both RESCALEs
                    # would bypass this clamp and cause up to 120 INT8 steps
                    # of output difference vs the reference model (D94483331).
                    continue
                else:
                    # Non-identity: replace with single INT32->INT32 RESCALE
                    with graph.inserting_before(user):
                        composed_node = create_node(
                            graph,
                            exir_ops.backend.tosa.RESCALE.default,
                            (
                                r1_input,
                                r2_output_dtype,
                                [composed_scale],
                                r1_input_zp,
                                r2_output_zp,
                            ),
                            from_node=user,
                        )
                    user.replace_all_uses_with(composed_node)
                    nodes_to_erase.append(user)

                modified = True

            # Always consider R1 for removal; actual erasure is guarded below
            nodes_to_erase.append(node)

        for node in nodes_to_erase:
            if len(node.users) == 0:
                graph.erase_node(node)

        if modified:
            graph_module.recompile()

        return PassResult(graph_module, modified)


def _is_rescale(node: Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == exir_ops.backend.tosa.RESCALE.default
    )
