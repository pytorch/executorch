# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

from typing import Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm._passes.decompose_embedding_pass import (
    DecomposeEmbeddingPass,
)
from executorch.backends.arm.tosa.specification import (
    context_supports_int64,
    get_context_spec,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import EdgeOpOverload, ExportPass, PassResult
from torch._subclasses.fake_tensor import FakeTensor


logger = logging.getLogger(__name__)


class InsertInt32CastsAfterInt64PlaceholdersPass(ArmPass):
    """Insert an int64->int32 cast after each int64 placeholder.

    Under PRO-INT with EXT-INT64 only int64 index arguments are cast to int32,
    and every other int64 value keeps int64.

    Note: Overflow checks are not applied in this pass. It is the user's responsibility to ensure that values fit within
    the int32 range.

    """

    _passes_required_after: Set[Type[ExportPass]] = {DecomposeEmbeddingPass}

    # Ops that require i64 inputs → positions of args to upcast.
    # Key: op overload; Value: zero-based indices of positional args that must be i64.
    I64_INPUT_ARG_POSITIONS = {
        torch.ops.aten.one_hot.default: (0,),
    }

    # Ops that take int32 indices → position of the index argument.
    INDEX_ARG_POSITIONS = {
        torch.ops.aten.embedding.default: 1,
        torch.ops.aten.gather.default: 2,
        torch.ops.aten.index.Tensor: 1,
        torch.ops.aten.index_put.default: 1,
        torch.ops.aten.index_put_.default: 1,
        torch.ops.aten.index_select.default: 2,
        torch.ops.aten.scatter.src: 2,
        torch.ops.aten.scatter.value: 2,
        torch.ops.aten.scatter_add.default: 2,
    }

    def _insert_callsite_i32_to_i64_casts(self, graph_module: torch.fx.GraphModule):
        """If an operator requires int64 inputs but dtype propagation (via
        call_operator) produced int32, insert a local int32→int64 cast at the
        call site to satisfy PyTorch's operator input validation.
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
            if node.op not in ("placeholder", "get_attr"):
                continue
            if "val" not in node.meta:
                continue  # Ignore submodule get_attrs
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
                self._placeholder_casts_inserted += 1
                modified = True
        return modified

    def _insert_index_i64_to_i32_casts(self, graph_module: torch.fx.GraphModule):
        modified = False
        graph = graph_module.graph
        casts: dict[torch.fx.Node, torch.fx.Node] = {}

        def to_int32(arg):
            if not isinstance(arg, torch.fx.Node) or not self._is_tensor_of_dtype(
                arg.meta.get("val"), torch.int64
            ):
                return arg
            if arg not in casts:
                with graph.inserting_after(arg):
                    casts[arg] = create_node(
                        graph,
                        self._get_decomposition(graph),
                        args=(arg,),
                        kwargs={"dtype": torch.int32},
                    )
            return casts[arg]

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            position = self.INDEX_ARG_POSITIONS.get(node.target)
            if position is None or len(node.args) <= position:
                continue
            index = node.args[position]
            indices = index if isinstance(index, (list, tuple)) else [index]
            new_indices = [to_int32(arg) for arg in indices]
            if any(old is not new for old, new in zip(indices, new_indices)):
                new_index = (
                    type(index)(new_indices)
                    if isinstance(index, (list, tuple))
                    else new_indices[0]
                )
                node.update_arg(position, new_index)
                modified = True
        self._placeholder_casts_inserted += len(casts)
        return modified

    def _insert_int64_promotion_casts(self, graph_module: torch.fx.GraphModule):
        """Cast int64 operands of operators that compute in a floating-point
        dtype to that dtype. When the spec has no floating-point profile, cast
        int64 operands of operators with a floating-point input to int32.
        """
        modified = False
        graph = graph_module.graph
        to_copy_op = self._get_decomposition(graph)
        support_float = get_context_spec().support_float()
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target in (
                to_copy_op,
                torch.ops.aten.to.dtype,
                torch.ops.aten.to.dtype_layout,
                torch.ops.aten._to_copy.default,
            ):
                continue
            input_dtypes = [
                value.dtype
                for value in (arg.meta.get("val") for arg in node.all_input_nodes)
                if isinstance(value, FakeTensor) and value.is_floating_point()
            ]
            output = node.meta.get("val")
            if not support_float:
                cast_dtypes = [torch.int32] if input_dtypes else []
            elif isinstance(output, FakeTensor) and output.is_floating_point():
                cast_dtypes = [output.dtype]
            else:
                cast_dtypes = input_dtypes
            if not cast_dtypes:
                continue
            for arg in node.all_input_nodes:
                if not self._is_tensor_of_dtype(arg.meta.get("val"), torch.int64):
                    continue
                with graph.inserting_before(node):
                    cast = create_node(
                        graph,
                        to_copy_op,
                        args=(arg,),
                        kwargs={"dtype": cast_dtypes[0]},
                    )
                node.replace_input_with(arg, cast)
                modified = True
        return modified

    def call(self, graph_module: torch.fx.GraphModule):
        self._placeholder_casts_inserted = 0
        modified = False
        if context_supports_int64():
            modified |= self._insert_index_i64_to_i32_casts(graph_module)
            modified |= self._insert_int64_promotion_casts(graph_module)
        else:
            modified |= self._insert_placeholder_i64_to_i32_casts(graph_module)
        modified |= self._insert_callsite_i32_to_i64_casts(graph_module)

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        if self._placeholder_casts_inserted:
            logger.warning(
                "InsertInt32CastsAfterInt64PlaceholdersPass: inserted %d "
                "int64-to-int32 placeholder cast(s).",
                self._placeholder_casts_inserted,
            )
        return PassResult(graph_module, modified)
