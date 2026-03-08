# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult


class EliminateRescaleBeforeMulPass(ArmPass):
    """Eliminate redundant INT32->INT32 RESCALE ops feeding exclusively into MUL.

    After InsertRescaleInt32Pass and FuseConsecutiveRescalesPass, the graph may
    contain INT32->INT32 RESCALE nodes between consecutive elementwise ops.
    When such a RESCALE feeds exclusively into MUL ops, it is computationally
    redundant and can be removed with a compensating scale adjustment on the
    downstream output RESCALE.

    Why only MUL (not ADD/SUB):
        For ADD/SUB, InsertRescaleInt32Pass rescales both inputs to a common
        scale (2 * max(lhs, rhs) / (1 << shift_bits)) to ensure correct
        integer arithmetic — the input RESCALE is required for operand
        alignment. For MUL, input scales remain unchanged because the output
        scale is the product of input scales (S_out = S_0 * S_1), regardless
        of what the input scales are. A RESCALE adjusting scale before MUL is
        therefore mathematically redundant: the adjustment can be absorbed
        into the downstream output RESCALE as
        new_out_scale = old_out_scale * removed_scale.
        See InsertRescaleInt32Pass._get_inputs_rescaled_qparams() for the
        scale arithmetic distinction.

    Why not Conv2D/MatMul boundaries:
        Empirically, eliminating RESCALE ops at Conv2D/MatMul boundaries
        causes the Vela NPU compiler to generate worse instruction schedules.
        The INT32->INT8->INT32 round-trips at those boundaries provide natural
        scheduling breaks that help Vela's register allocator. Removing them
        caused +12.9% (CC) and +16.1% (Detector) cycle regressions.

    When multiple eligible RESCALEs feed the same MUL (e.g., both inputs have
    INT32->INT32 RESCALEs), each is eliminated sequentially. The downstream
    scale adjustments compose correctly because MUL's output scale is
    multiplicative: removing RESCALE_A (scale S_a) then RESCALE_B (scale S_b)
    yields new_out_scale = old_out_scale * S_a * S_b, which is correct.
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

            # Must be INT32 output
            if node.args[1] != torch.int32:
                continue

            # Must have zero points of 0 (INT32->INT32 rescales from
            # InsertRescaleInt32Pass always have zp=0)
            input_zp = node.args[3]
            output_zp = node.args[4]
            if input_zp != 0 or output_zp != 0:
                continue

            # All users must be MUL ops
            if len(node.users) == 0:
                continue
            if not all(
                u.op == "call_function"
                and u.target == exir_ops.edge.aten.mul.Tensor
                for u in node.users
            ):
                continue

            # All downstream users of each MUL must be RESCALEs so we can
            # compensate for the removed scale. Without this guard, non-RESCALE
            # consumers of MUL would receive incorrectly scaled values.
            if not all(
                mul_out.users and all(_is_rescale(u) for u in mul_out.users)
                for mul_out in node.users
            ):
                continue

            # All downstream RESCALEs must produce INT32 (staying within the
            # INT32 computation region). If any converts to INT8/INT16, it
            # defines a quantization boundary where the annotated scale must
            # match the actual integer values. Modifying such a RESCALE would
            # break TABLE ops (exp, log, sigmoid, etc.) that build lookup
            # tables from the quantization annotation, and would also affect
            # Conv/MatMul boundaries where Vela relies on precise scaling.
            if not all(
                mul_output_user.args[1] == torch.int32
                for mul_out in node.users
                for mul_output_user in mul_out.users
            ):
                continue

            # Check that the input is also INT32 — the preceding node should
            # produce INT32 (either another RESCALE with INT32 output, or an
            # elementwise op wrapped by InsertRescaleInt32Pass).
            rescale_input = node.args[0]
            if not _produces_int32(rescale_input):
                continue

            removed_scale = float(node.args[2][0])

            # Adjust the downstream output RESCALE scale for each MUL user
            for mul_user in list(node.users):
                for mul_output_user in list(mul_user.users):
                    old_scale = float(mul_output_user.args[2][0])
                    new_scale = old_scale * removed_scale
                    args = list(mul_output_user.args)
                    args[2] = [new_scale]
                    mul_output_user.args = tuple(args)

            # Replace the RESCALE with its input
            node.replace_all_uses_with(rescale_input)
            nodes_to_erase.append(node)
            modified = True

        for n in nodes_to_erase:
            if len(n.users) == 0:
                graph.erase_node(n)

        if modified:
            graph_module = super().call(graph_module).graph_module
            graph_module.recompile()

        return PassResult(graph_module, modified)


def _is_rescale(node: Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == exir_ops.backend.tosa.RESCALE.default
    )


def _produces_int32(node: Node) -> bool:
    """Check if a node produces INT32 output."""
    if isinstance(node, Node):
        # If it's a RESCALE, check its output dtype arg
        if _is_rescale(node):
            return node.args[1] == torch.int32
        # For other ops, check the fake tensor metadata
        if "val" in node.meta:
            val = node.meta["val"]
            if isinstance(val, torch.Tensor) and val.dtype == torch.int32:
                return True
            if hasattr(val, "dtype") and val.dtype == torch.int32:
                return True
    return False
