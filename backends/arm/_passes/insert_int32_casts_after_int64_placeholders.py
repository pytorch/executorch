# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import logging

import torch
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import EdgeOpOverload, ExportPass, PassResult
from torch._subclasses.fake_tensor import FakeTensor


logger = logging.getLogger(__name__)


class InsertInt32CastsAfterInt64PlaceholdersPass(ExportPass):
    """
    Insert an int64->int32 cast after each int64 placeholder.

    Note: Overflow checks are not applied in this pass. It is the user's responsibility to ensure that values fit within
    the int32 range.
    """

    # Ops that require i64 inputs → positions of args to upcast.
    # Key: op overload; Value: zero-based indices of positional args that must be i64.
    I64_INPUT_ARG_POSITIONS = {
        torch.ops.aten.one_hot.default: (0,),
    }

    def _insert_callsite_i32_to_i64_casts(self, graph_module: torch.fx.GraphModule):
        """
        If an operator requires int64 inputs but dtype propagation (via call_operator)
        produced int32, insert a local int32→int64 cast at the call site to satisfy
        PyTorch's operator input validation.
        """
        modified = False
        graph = graph_module.graph
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target not in self.I64_INPUT_ARG_POSITIONS:
                continue

            with graph.inserting_before(node):
                arg_positions = self.I64_INPUT_ARG_POSITIONS.get(node.target)
                args_list = list(node.args)
                for pos in arg_positions:  # type: ignore[union-attr]
                    input_arg = args_list[pos]
                    to_copy_op = self._get_decomposition(graph)
                    cast_node = graph_module.graph.create_node(
                        "call_function",
                        to_copy_op,
                        (input_arg,),
                        {"dtype": torch.int64},
                    )
                    cast_node.meta["val"] = node.meta["val"].to(torch.int64)
                    args_list[pos] = cast_node
                node.args = tuple(args_list)
                modified = True
        return modified

    def _graph_uses_edge_ops(self, graph: torch.fx.Graph) -> bool:
        for n in graph.nodes:
            if n.op == "call_function":
                if isinstance(n.target, EdgeOpOverload):
                    return True
        return False

    def _get_decomposition(self, graph: torch.fx.Graph):
        if self._graph_uses_edge_ops(graph):
            return exir_ops.edge.dim_order_ops._to_dim_order_copy.default
        else:
            return torch.ops.dim_order_ops._to_dim_order_copy.default

    def _is_tensor_of_dtype(self, node_val, dtype: torch.dtype) -> bool:
        return isinstance(node_val, FakeTensor) and node_val.dtype == dtype

    def _insert_placeholder_i64_to_i32_casts(self, graph_module: torch.fx.GraphModule):
        modified = False
        graph = graph_module.graph
        for node in graph.nodes:
            if node.op != "placeholder":
                continue
            node_val = node.meta["val"]
            if not self._is_tensor_of_dtype(node_val, torch.int64):
                continue

            to_copy_op = self._get_decomposition(graph)
            with graph.inserting_after(node):
                cast_after = create_node(
                    graph,
                    to_copy_op,
                    args=(node,),
                    kwargs={
                        "dtype": torch.int32,
                    },
                )
                users = [user for user in node.users if user != cast_after]
                for user in users:
                    user.replace_input_with(node, cast_after)
                logger.warning(
                    f"Inserting a casting node {cast_after.name} after {node.name} to cast int64 placeholder"
                    f" to int32 for {node.name} defined in {node.meta.get('stack_trace','[no stack trace found]')}"
                )
                modified = True
        return modified

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        modified |= self._insert_placeholder_i64_to_i32_casts(graph_module)
        modified |= self._insert_callsite_i32_to_i64_casts(graph_module)

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified)
