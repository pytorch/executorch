# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass_utils import create_node, get_node_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

DTYPE_RANK = {
    torch.bool: 0,
    torch.uint8: 1,
    torch.int8: 2,
    torch.int16: 3,
    torch.int32: 4,
    torch.int64: 5,
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
}


def get_largest_dtype(dtype_1, dtype_2):
    """Find the largest dtype."""
    return dtype_1 if DTYPE_RANK[dtype_1] > DTYPE_RANK[dtype_2] else dtype_2


class MatchArgDtypePass(ExportPass):
    """Pass to match data types of non-condition input tensors.

    Edge dialect allows different data types for non-condition tensors, while TOSA
    does not. In cases where they differ a TOSA CAST operator is inserted.

    There is an edge case where one input is `boolean`, which cannot be directly cast
    to, for example, float32. When this occurs two CAST operators are added to first
    cast to int8 and then to the correct target data type.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    targeted_ops = {exir_ops.edge.aten.sub.Tensor, exir_ops.edge.aten.where.self}

    def call(self, graph_module: torch.fx.GraphModule):
        modified_graph = False
        graph = graph_module.graph

        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in self.targeted_ops:
                continue

            input_ = get_node_arg(node.args, 0)
            other_ = get_node_arg(node.args, 1)

            input_dtype = input_.meta["val"].dtype
            other_dtype = other_.meta["val"].dtype
            target_dtype = input_dtype
            if input_dtype != other_dtype:
                target_dtype = get_largest_dtype(input_dtype, other_dtype)

            for arg in node.args[1:]:
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
                            replace_node_fp32 = create_node(
                                graph,
                                exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                            )
                            replace_node_fp32.args = (replace_node_int8,)
                            replace_node_fp32.kwargs = {"dtype": target_dtype}
                            node.replace_input_with(arg, replace_node_fp32)
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
