# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import logging
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
    set_node_arg,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


logger = logging.getLogger(__name__)


class ConvertInt64OutputOpsToInt32Pass(ArmPass):
    """
    Rewrites or removes operations that produce int64 outputs, converting them
    to int32 where possible.


    Currently, this pass handles casting and argmax operators:
      1. int32 -> int64:
         removes the cast and redirects all uses to the original int32 value.
      2. other types -> int64:
         rewrites the cast to produce int32 instead of int64.
      3. torch.argmax()
         insert an int64->int32 cast after the argmax node

    Future extensions may include operators that return int64 outputs by default
    (e.g., `argmin`), rewriting them or inserting an int64 -> int32 cast to yield
    int32 results.

    Note: Overflow checks are applied selectively in this pass. For operators without
    such checks, it is the user's responsibility to ensure that values fit within
    the int32 range.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    aten_cast_ops = (
        torch.ops.aten.to.dtype,
        torch.ops.aten.to.dtype_layout,
    )
    edge_cast_ops = (exir_ops.edge.dim_order_ops._to_dim_order_copy.default,)

    aten_argmax_ops = (torch.ops.aten.argmax.default,)
    edge_argmax_ops = (exir_ops.edge.aten.argmax.default,)

    aten_ops = aten_cast_ops + aten_argmax_ops
    edge_ops = edge_cast_ops + edge_argmax_ops

    # dtype is specified in args
    cast_ops_args = (
        torch.ops.aten.to.dtype,  # to_2: node.args: (gt, torch.int64) node.kwargs: {}
    )
    # dtype is specified in kwargs
    cast_ops_kwargs = (
        torch.ops.aten.to.dtype_layout,  # to_1: node.args: (unsqueeze,) node.kwargs: {'dtype': torch.int64, 'layout': torch.strided, 'device': device(type='cpu')}
        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,  # node.args: (aten_gt_scalar,) node.kwargs: {'dtype': torch.int64, 'dim_order': [0, 1]}
    )

    def _get_decomposition(self, op):
        if op in self.edge_ops:
            return exir_ops.edge.dim_order_ops._to_dim_order_copy.default

        if op in self.aten_ops:
            return torch.ops.dim_order_ops._to_dim_order_copy.default

        raise RuntimeError(
            f"[{self.__class__.__name__}] Can't get decomposition for op {op}"
        )

    def _convert_casting_operators(self, node: torch.fx.Node):
        input_node = node.all_input_nodes[0]
        input_dtype = get_first_fake_tensor(input_node).dtype
        # Case 1: int32 -> int64 - removes the ops
        if input_dtype == torch.int32:
            users = [user for user in node.users if node != user]
            for user in users:
                logger.warning(
                    f"Removing int32->int64 casting node {node.name} defined in"
                    f" {node.meta.get('stack_trace','[no stack trace found]')}"
                )
                user.replace_input_with(node, input_node)
        # Case 2: other types -> int64 - rewrites to cast to int32
        else:
            if node.target in self.cast_ops_kwargs:
                set_node_arg(node, "dtype", torch.int32)
            elif node.target in self.cast_ops_args:
                set_node_arg(node, 1, torch.int32)
            else:
                raise RuntimeError(f"Unexpected target {node.target} in {node.name}")
            output_dtype = get_first_fake_tensor(node).dtype
            logger.warning(
                f"Converting casting node {node.name} from {input_dtype}->{output_dtype} to"
                f" {input_dtype}->torch.int32 defined in {node.meta.get('stack_trace','[no stack trace found]')}"
            )

    def _convert_argmax_operators(self, node: torch.fx.Node, graph: torch.fx.Graph):
        output_tensor = node
        to_copy_op = self._get_decomposition(node.target)
        with graph.inserting_after(node):
            cast_after = create_node(
                graph,
                to_copy_op,
                args=(output_tensor,),
                kwargs={
                    "dtype": torch.int32,
                },
            )
            users = [user for user in node.users if user != cast_after]
            for user in users:
                user.replace_input_with(output_tensor, cast_after)
            logger.warning(
                f"Inserting a casting node {cast_after.name} after {node.name} to cast int64 output"
                f" to int32 for {node.name} defined in {node.meta.get('stack_trace','[no stack trace found]')}"
            )

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target not in self.aten_ops + self.edge_ops:
                continue
            output_dtype = get_first_fake_tensor(node).dtype
            if output_dtype != torch.int64:
                continue

            if node.target in self.aten_cast_ops + self.edge_cast_ops:
                self._convert_casting_operators(node)
            elif node.target in self.aten_argmax_ops + self.edge_argmax_ops:
                # TODO: Add range check based on the input tensor shape before casting the output
                self._convert_argmax_operators(node, graph)
            else:
                raise RuntimeError(f"Unexpected target {node.target} in {node.name}")

            modified = True

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
