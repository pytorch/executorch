# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm.constants import DQ_OPS
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class ConvertELUParamsPass(ArmPass):
    """The int8 ELU operator crashes when the alpha, scale or input scale
    parameters are not integers.

    This pass temporarily converts quantized ELU parameters to int and stores
    the original float values in the meta dict to be able to recover them later.

    """

    @property
    def _passes_required_after(self) -> Set[Type[ExportPass]]:
        # Lazy import to avoid circular dependency between passes
        from executorch.backends.arm._passes.insert_table_ops import InsertTableOpsPass

        return {InsertTableOpsPass}

    def call(self, graph_module: torch.fx.GraphModule):
        modified_graph = False
        graph = graph_module.graph
        node_list = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.elu.default
        )
        for node in node_list:
            input_node = node.all_input_nodes[0]
            is_quantized = (
                input_node.op == "call_function" and input_node.target in DQ_OPS
            )
            if not is_quantized or not self.allowed_to_transform(node.meta):
                continue

            with graph.inserting_after(node):
                replace_node = create_node(
                    graph, exir_ops.edge.aten.elu.default, from_node=node
                )

                old_args = list(node.args)
                alpha = (
                    old_args[1] if len(old_args) > 1 else node.kwargs.get("alpha", 1.0)
                )
                scale = (
                    old_args[2] if len(old_args) > 2 else node.kwargs.get("scale", 1.0)
                )
                input_scale = (
                    old_args[3]
                    if len(old_args) > 3
                    else node.kwargs.get("input_scale", 1.0)
                )

                replace_node.args = (old_args[0],)

                # Set placeholder int values
                updated_kwargs = dict(node.kwargs)
                updated_kwargs["alpha"] = 1
                updated_kwargs["scale"] = 1
                updated_kwargs["input_scale"] = (
                    2  # Keep input_scale away from 1 to avoid fake execution type checks.
                )
                replace_node.kwargs = updated_kwargs

                # Save correct parameters
                replace_node.meta["float_alpha"] = alpha
                replace_node.meta["float_scale"] = scale
                replace_node.meta["float_input_scale"] = input_scale

                node.replace_all_uses_with(replace_node)
                graph.erase_node(node)
                modified_graph = True

        if modified_graph:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified_graph)
