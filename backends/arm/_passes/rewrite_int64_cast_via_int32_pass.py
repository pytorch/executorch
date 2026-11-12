# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch

from executorch.backends.arm._passes.arm_pass import ArmOpTargetedPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
    set_node_arg,
)
from executorch.backends.arm.tosa.specification import context_supports_int64
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RewriteInt64CastViaInt32Pass(ArmOpTargetedPass):
    """Rewrite to_dim_order_copy casts between int64 and int8 or int16 into two
    casts through int32.

    TOSA has no direct cast between int64 and int8 or int16, while each cast to
    and from int32 is supported.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    target_ops = {exir_ops.edge.dim_order_ops._to_dim_order_copy.default}

    _via_int32 = ({torch.int8, torch.int64}, {torch.int16, torch.int64})

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False

        if not context_supports_int64():
            return PassResult(graph_module, modified)

        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in self.target_ops:
                continue

            input_dtype = get_first_fake_tensor(node.all_input_nodes[0]).dtype
            output_dtype = get_first_fake_tensor(node).dtype
            if {input_dtype, output_dtype} not in self._via_int32:
                continue

            set_node_arg(node, "dtype", torch.int32)

            users = list(node.users)
            with graph.inserting_after(node):
                cast_after = create_node(
                    graph,
                    node.target,
                    args=(node,),
                    kwargs={"dtype": output_dtype},
                )
            for user in users:
                user.replace_input_with(node, cast_after)
            modified = True

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
