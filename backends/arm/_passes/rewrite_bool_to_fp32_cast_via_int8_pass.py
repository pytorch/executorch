# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch

from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
    set_node_arg,
)
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RewriteBoolToFp32CastViaInt8Pass(ArmPass):
    """
    Legalizes unsupported bool->fp32 to_dim_order_copy casts for the Arm TOSA
    backend when both integer and float TOSA profiles are enabled.

    For the combined INT+FP profile, this pass rewrites a single bool->fp32 cast
    into a bool->int8 cast followed by an int8->fp32 cast, so that each cast
    is individually supported by the TOSA INT and FP profiles. For other
    profiles (INT-only or FP-only) the pass is a no-op.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    targeted_ops = {exir_ops.edge.dim_order_ops._to_dim_order_copy.default}

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False

        tosa_spec = get_context_spec()
        if not (tosa_spec.support_integer() and tosa_spec.support_float()):
            return PassResult(graph_module, modified)

        graph = graph_module.graph
        for node in graph.nodes:
            if node.op != "call_function" or node.target not in self.targeted_ops:
                continue

            input_node = node.all_input_nodes[0]
            input_dtype = get_first_fake_tensor(input_node).dtype
            if input_dtype != torch.bool:
                continue

            output_dtype = get_first_fake_tensor(node).dtype
            if output_dtype != torch.float32:
                continue

            set_node_arg(node, "dtype", torch.int8)

            users = list(node.users)
            with graph.inserting_after(node):
                cast_after = create_node(
                    graph,
                    node.target,
                    args=(node,),
                    kwargs={
                        "dtype": torch.float32,
                    },
                )
                for user in users:
                    user.replace_input_with(node, cast_after)
                modified = True

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
