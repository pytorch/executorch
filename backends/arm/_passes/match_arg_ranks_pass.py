# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast, Set, Type

from executorch.backends.arm._passes import ArmPass

from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)
from executorch.exir import ExportedProgram

from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class MatchArgRanksPass(ArmPass):
    """
    For ops in 'targeted_ops', make sure that the inputs share the same rank.
    New dimensions are inserted from the beginning of the inputs that have a
    lower rank to match the input with the highest rank.

    Example:
        input0 = shape(4, 3, 2)
        input1 = shape(2)
        input2 = shape(3, 1)
    Becomes:
        input0 = shape(4, 3, 2)
        input1 = shape(1, 1, 2)
        input2 = shape(1, 3, 1)
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.exported_program = exported_program

    targeted_ops = [
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.div.Tensor,
        exir_ops.edge.aten.bitwise_right_shift.Tensor,
        exir_ops.edge.aten.bitwise_left_shift.Tensor,
        exir_ops.edge.aten.eq.Tensor,
        exir_ops.edge.aten.gt.Tensor,
        exir_ops.edge.aten.ge.Tensor,
        exir_ops.edge.aten.lt.Tensor,
        exir_ops.edge.aten.le.Tensor,
        exir_ops.edge.aten.pow.Tensor_Tensor,
        exir_ops.edge.aten.where.self,
        exir_ops.edge.aten.bitwise_and.Tensor,
        exir_ops.edge.aten.bitwise_xor.Tensor,
        exir_ops.edge.aten.bitwise_or.Tensor,
    ]

    def _match_op_rank(self, graph_module, node, arg, max_rank):
        """
        In graph_module, insert a view between arg and node to make the
        rank of arg match the other args to node.
        """
        shape = get_first_fake_tensor(arg).shape
        rank = len(shape)
        new_shape = list([1] * (max_rank - rank) + list(shape))
        with graph_module.graph.inserting_before(node):
            view = create_node(
                graph_module.graph,
                exir_ops.edge.aten.view_copy.default,
                args=(arg, new_shape),
                kwargs={},
            )
            node.replace_input_with(arg, view)

    def call(self, graph_module: GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            node = cast(Node, node)

            if node.op != "call_function" or node.target not in self.targeted_ops:
                continue

            # Calculate max rank of all inputs to node
            max_rank = 0
            for arg in node.args:
                if isinstance(arg, Node):
                    shape = get_first_fake_tensor(arg).shape
                    max_rank = max(max_rank, len(shape))

            # Adjust output shape of args if needed.
            for arg in node.args:
                if not isinstance(arg, Node):
                    continue
                shape = get_first_fake_tensor(arg).shape
                rank = len(shape)
                if rank == max_rank:
                    continue

                self._match_op_rank(graph_module, node, arg, max_rank)

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
