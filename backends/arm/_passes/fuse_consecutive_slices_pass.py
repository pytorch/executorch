# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence, Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import create_node
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node

_SLICE_INPUT = 0
_SLICE_START = 1
_SLICE_SIZE = 2


class FuseConsecutiveSlicesPass(ArmPass):
    """Fuse consecutive ``tosa.SLICE`` operations into one ``tosa.SLICE``.

    For static slice operands, ``slice(slice(x, start_1, size_1), start_2,
    size_2)`` is equivalent to ``slice(x, start_1 + start_2, size_2)``. The
    pass rewrites the second slice to use the original input and removes the
    first slice when it becomes dead. Dynamic shape operands are left unchanged.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False

        for node in list(graph.nodes):
            if self._try_fuse_slice(node):
                modified = True

        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, modified)

    @staticmethod
    def _is_slice(node: Node) -> bool:
        return (
            node.op == "call_function"
            and node.target == exir_ops.backend.tosa.SLICE.default
        )

    @staticmethod
    def _constant_shape(shape_arg) -> list[int] | None:
        if isinstance(shape_arg, Node):
            if shape_arg.target != exir_ops.backend.tosa.CONST_SHAPE.default:
                return None
            shape_arg = shape_arg.args[0]

        if not isinstance(shape_arg, Sequence) or isinstance(shape_arg, str):
            return None
        if not all(isinstance(dim, int) for dim in shape_arg):
            return None
        return list(shape_arg)

    @staticmethod
    def _create_const_shape(graph: torch.fx.Graph, shape: list[int]) -> Node:
        node = graph.create_node(
            "call_function",
            exir_ops.backend.tosa.CONST_SHAPE.default,
            args=(shape,),
        )
        node.meta = {
            "val": shape,
            TosaSpecialDtype.meta_key(): TosaSpecialDtype.SHAPE,
        }
        return node

    def _try_fuse_slice(self, node: Node) -> bool:
        if not self._is_slice(node):
            return False

        input_node = node.args[_SLICE_INPUT]
        if not isinstance(input_node, Node) or not self._is_slice(input_node):
            return False

        first_start = self._constant_shape(input_node.args[_SLICE_START])
        second_start = self._constant_shape(node.args[_SLICE_START])
        second_size = self._constant_shape(node.args[_SLICE_SIZE])
        if first_start is None or second_start is None or second_size is None:
            return False
        if not (len(first_start) == len(second_start) == len(second_size)):
            return False

        fused_start = [outer + inner for outer, inner in zip(first_start, second_start)]
        graph = node.graph

        with graph.inserting_before(node):
            fused_start_node = self._create_const_shape(graph, fused_start)
            fused_size_node = self._create_const_shape(graph, second_size)
            fused_slice = create_node(
                graph,
                exir_ops.backend.tosa.SLICE.default,
                args=(input_node.args[_SLICE_INPUT], fused_start_node, fused_size_node),
                from_node=node,
                inherit_qparams=True,
            )

        node.replace_all_uses_with(fused_slice)
        return True
