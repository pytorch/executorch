# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class MatchArgDtypePass(ArmOpTargetedPass):
    """Pass to match data types of non-condition input tensors.

    Edge dialect allows different data types for non-condition tensors, while TOSA
    does not. In cases where they differ a TOSA CAST operator is inserted.

    There is an edge case where one input is `boolean`, which cannot be directly cast
    to, for example, float32. When this occurs two CAST operators are added to first
    cast to int8 and then to the correct target data type.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    target_ops = {
        exir_ops.edge.aten.maximum.default,
        exir_ops.edge.aten.pow.Tensor_Tensor,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.where.self,
    }

    def call(self, graph_module: torch.fx.GraphModule):
        modified_graph = False
        graph = graph_module.graph

        for node in graph.nodes:
            if node.op != "call_function" or node.target not in self.target_ops:
                continue
            args_to_cast = (
                node.args[1:]
                if node.target == exir_ops.edge.aten.where.self
                else node.args
            )
            input_ = args_to_cast[0]
            other_ = args_to_cast[1]
            target_dtype = torch.result_type(input_.meta["val"], other_.meta["val"])

            for arg in args_to_cast:
                arg_dtype = arg.meta["val"].dtype

                if arg_dtype != target_dtype:
                    if arg_dtype == torch.bool:
                        # Bool is an edge case which cannot necessarily be directly
                        # converted to the target data type.
                        with graph.inserting_after(arg):
                            replace_node_int8 = create_node(
                                graph,
                                exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                            )
                            replace_node_int8.args = (arg,)
                            replace_node_int8.kwargs = {"dtype": torch.int8}

                        with graph.inserting_after(replace_node_int8):
                            replace_node_target = create_node(
                                graph,
                                exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                            )
                            replace_node_target.args = (replace_node_int8,)
                            replace_node_target.kwargs = {"dtype": target_dtype}
                            node.replace_input_with(arg, replace_node_target)
                    else:
                        with graph.inserting_after(arg):
                            replace_node = create_node(
                                graph,
                                exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                            )
                            replace_node.args = (arg,)
                            replace_node.kwargs = {"dtype": target_dtype}
                            node.replace_input_with(arg, replace_node)

                    modified_graph = True

        if modified_graph:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified_graph)
