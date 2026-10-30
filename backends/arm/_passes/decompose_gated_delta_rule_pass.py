# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, PassResult


class DecomposeGatedDeltaRulePass(ArmPass):
    """Unroll the scalar or channelwise gated delta recurrence for TOSA."""

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if "gated_delta_rule" not in str(node.target):
                continue
            target = torch.ops.llama.gated_delta_rule.default
            if node.target != target and getattr(node.target, "_op", None) != target:
                continue
            aten = (
                exir_ops.edge.aten
                if isinstance(node.target, EdgeOpOverload)
                else torch.ops.aten
            )

            query, key, value, decay, beta, state = node.args
            query_shape = query.meta["val"].shape
            key_shape = key.meta["val"].shape
            value_shape = value.meta["val"].shape
            decay_shape = decay.meta["val"].shape
            batch, heads, seq_len, key_dim = key_shape
            value_dim = value_shape[-1]

            # The grouped-head case needs a separate output projection. A
            # symbolic sequence length cannot be unrolled into a static graph.
            if (
                query_shape[1] != heads
                or not isinstance(seq_len, int)
                or seq_len < 1
                or any(user.target != operator.getitem for user in node.users)
            ):
                continue

            def emit(op, *args, from_node=node):
                return create_node(graph, op, args=args, from_node=from_node)

            outputs = []
            with graph.inserting_before(node):
                for index in range(seq_len):
                    q_t = emit(aten.select_copy.int, query, 2, index)
                    k_t = emit(aten.select_copy.int, key, 2, index)
                    v_t = emit(aten.select_copy.int, value, 2, index)
                    d_t = emit(aten.select_copy.int, decay, 2, index)
                    b_t = emit(aten.select_copy.int, beta, 2, index)

                    d_t = emit(aten.unsqueeze_copy.default, d_t, -1)
                    if len(decay_shape) == 3:
                        d_t = emit(aten.unsqueeze_copy.default, d_t, -1)
                    state = emit(aten.mul.Tensor, state, d_t)

                    k_mat = emit(
                        aten.view_copy.default,
                        k_t,
                        [batch * heads, 1, key_dim],
                    )
                    state_mat = emit(
                        aten.view_copy.default,
                        state,
                        [batch * heads, key_dim, value_dim],
                    )
                    predicted = emit(aten.bmm.default, k_mat, state_mat)
                    predicted = emit(
                        aten.view_copy.default,
                        predicted,
                        [batch, heads, value_dim],
                    )
                    delta = emit(aten.sub.Tensor, v_t, predicted)
                    b_t = emit(aten.unsqueeze_copy.default, b_t, -1)
                    delta = emit(aten.mul.Tensor, delta, b_t)
                    k_t = emit(aten.unsqueeze_copy.default, k_t, -1)
                    delta = emit(aten.unsqueeze_copy.default, delta, -2)
                    update = emit(aten.mul.Tensor, k_t, delta)
                    state = emit(aten.add.Tensor, state, update)

                    q_mat = emit(
                        aten.view_copy.default,
                        q_t,
                        [batch * heads, 1, key_dim],
                    )
                    state_mat = emit(
                        aten.view_copy.default,
                        state,
                        [batch * heads, key_dim, value_dim],
                    )
                    output = emit(aten.bmm.default, q_mat, state_mat)
                    output = emit(
                        aten.view_copy.default,
                        output,
                        [batch, heads, 1, value_dim],
                    )
                    outputs.append(output)

                output = emit(aten.cat.default, outputs, 2)

            for user in list(node.users):
                if user.target != operator.getitem:
                    continue
                replacement = output if user.args[1] == 0 else state
                user.replace_all_uses_with(replacement)
                graph.erase_node(user)
            graph.erase_node(node)
            modified = True

        if modified:
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
