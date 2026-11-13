# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Any, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.cast_int_comparison_inputs_pass import (
    CastIntComparisonInputsPass,
)
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch._export.utils import is_buffer, is_lifted_tensor_constant
from torch.export import ExportedProgram

logger = logging.getLogger(__name__)


class CastInt64BuffersToInt32Pass(ArmPass):
    """Cast int64 buffers and lifted tensor constants to int32 when in range.

    With the int64 extension under PRO-INT, a buffer whose users all compute in
    int64, compare it or return it keeps int64. Otherwise those users get the
    narrowed buffer cast back to int64. A lifted tensor constant used as an
    index is narrowed the same way.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _comparison_ops = CastIntComparisonInputsPass.target_ops | {
        exir_ops.edge.aten.eq.Scalar,
        exir_ops.edge.aten.ne.Scalar,
        exir_ops.edge.aten.ge.Scalar,
        exir_ops.edge.aten.gt.Scalar,
        exir_ops.edge.aten.le.Scalar,
        exir_ops.edge.aten.lt.Scalar,
    }

    def __init__(self, exported_program: ExportedProgram, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    @staticmethod
    def _is_index_use(node: torch.fx.Node, user: torch.fx.Node) -> bool:
        """Return True when ``user`` uses ``node`` only as an index tensor."""
        index_ops = {
            torch.ops.aten.index.Tensor,
            exir_ops.edge.aten.index.Tensor,
        }

        def contains_node(value: object) -> bool:
            if value is node:
                return True
            if isinstance(value, (tuple, list)):
                return any(contains_node(item) for item in value)
            if isinstance(value, dict):
                return any(contains_node(item) for item in value.values())
            if isinstance(value, slice):
                return any(
                    contains_node(item)
                    for item in (value.start, value.stop, value.step)
                )
            return False

        if (
            user.op != "call_function"
            or user.target not in index_ops
            or len(user.args) < 2
            or not isinstance(user.args[1], (list, tuple))
            or not any(index is node for index in user.args[1])
        ):
            return False

        # args[1] is the index-list operand. All other uses are unsafe.
        return not (
            any(
                contains_node(argument)
                for i, argument in enumerate(user.args)
                if i != 1
            )
            or contains_node(user.kwargs)
        )

    @classmethod
    def _is_index_tensor_input(cls, node: torch.fx.Node) -> bool:
        """Return True only when a constant is used exclusively as an index.

        ``node.users`` tracks consumer operations, not operand occurrences.
        A consumer can use this node as both indexed data and an index.

        """
        return bool(node.users) and all(
            cls._is_index_use(node, user) for user in node.users
        )

    def _is_cast_candidate(self, node: torch.fx.Node) -> bool:
        """Select int64 buffers and lifted tensor constants used as an index."""
        if not node.users:
            return False
        fake_tensor = node.meta.get("val")
        if not isinstance(fake_tensor, torch._subclasses.fake_tensor.FakeTensor):
            return False
        if fake_tensor.dtype != torch.int64:
            return False
        if is_buffer(self.exported_program, node):
            return True
        if not is_lifted_tensor_constant(self.exported_program, node):
            return False
        if self._is_index_tensor_input(node):
            return True
        return (
            any(self._is_index_use(node, user) for user in node.users)
            and self._supports_int64()
        )

    def should_run_pass(self, graph_module: torch.fx.GraphModule) -> bool:
        return any(self._is_cast_candidate(node) for node in graph_module.graph.nodes)

    def _assert_within_int32(self, tensor: torch.Tensor, node: torch.fx.Node):
        # An empty tensor has no values outside the int32 range.
        if tensor.numel() == 0:
            return

        if torch.min(tensor) < torch.iinfo(torch.int32).min:
            raise RuntimeError(
                f"Node {node.name} has value < {torch.iinfo(torch.int32).min}"
            )
        if torch.max(tensor) > torch.iinfo(torch.int32).max:
            raise RuntimeError(
                f"Node {node.name} has value > {torch.iinfo(torch.int32).max}"
            )

    def _cast_constant_to_int32(self, node: torch.fx.Node) -> bool:
        signature = self.exported_program.graph_signature
        if is_buffer(self.exported_program, node):
            constant_name = signature.inputs_to_buffers[node.name]
            # Persistent buffers live in state_dict, non-persistent ones in constants.
            store = (
                self.exported_program.state_dict
                if constant_name in self.exported_program.state_dict
                else self.exported_program.constants
            )
        elif is_lifted_tensor_constant(self.exported_program, node):
            constant_name = signature.inputs_to_lifted_tensor_constants[node.name]
            store = self.exported_program.constants
        else:
            return False

        if constant_name not in store:
            logger.warning(
                "Skipping int64 constant %s (%s): value not found in its storage",
                node.name,
                constant_name,
            )
            return False

        value = store[constant_name]
        # Do not truncate values outside the signed int32 range.
        self._assert_within_int32(value, node)
        store[constant_name] = value.to(torch.int32)
        node.meta["val"] = node.meta["val"].to(torch.int32)
        return True

    @classmethod
    def _compares_in_int64(cls, user: torch.fx.Node) -> bool:
        if user.target not in cls._comparison_ops:
            return False
        operands: list[Any] = [
            arg.meta["val"] if isinstance(arg, torch.fx.Node) else arg
            for arg in user.args[:2]
        ]
        return torch.result_type(*operands) == torch.int64 and all(
            arg.meta["val"].dtype in (torch.int64, torch.int32, torch.bool)
            for arg in user.all_input_nodes
            if isinstance(arg.meta.get("val"), torch.Tensor)
        )

    @classmethod
    def _computes_in_int64(cls, user: torch.fx.Node) -> bool:
        value = user.meta.get("val")
        return (
            user.op == "output"
            or cls._compares_in_int64(user)
            or (isinstance(value, torch.Tensor) and value.dtype == torch.int64)
        )

    @staticmethod
    def _supports_int64() -> bool:
        tosa_spec = get_context_spec()
        return tosa_spec.support_integer() and tosa_spec.support_extension("int64")

    @staticmethod
    def _cast_input_to_int64(
        graph: torch.fx.Graph, node: torch.fx.Node, user: torch.fx.Node
    ):
        with graph.inserting_before(user):
            cast = graph.call_function(
                exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                (node,),
                {"dtype": torch.int64},
            )
        cast.meta["val"] = node.meta["val"].to(torch.int64)
        user.replace_input_with(node, cast)

    def _to_int32(self, graph_module: torch.fx.GraphModule) -> int:
        converted = 0
        for node in list(graph_module.graph.nodes):
            if not self._is_cast_candidate(node):
                continue
            int64_users = []
            if not self._is_index_tensor_input(node) and self._supports_int64():
                int64_users = [
                    user for user in node.users if self._computes_in_int64(user)
                ]
                if len(int64_users) == len(node.users):
                    continue
            if self._cast_constant_to_int32(node):
                converted += 1
                for user in int64_users:
                    self._cast_input_to_int64(graph_module.graph, node, user)
        return converted

    def call(self, graph_module: torch.fx.GraphModule):
        converted = self._to_int32(graph_module)
        modified = converted > 0
        if modified:
            graph_module = super().call(graph_module).graph_module
            logger.warning(
                "CastInt64BuffersToInt32Pass: cast %d int64 buffer(s) or "
                "lifted tensor constant(s) to int32.",
                converted,
            )
        return PassResult(graph_module, modified)
