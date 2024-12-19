# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast

from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)

from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class MatchArgRanksPass(ExportPass):
    """
    For ops in 'targeted_ops', make sure that the inputs share the same rank.
    New dimensions are inserted at from the beginning of the
    """

    def __init__(self, exported_program):
        super().__init__()
        self.exported_program = exported_program

    targeted_ops = [
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.div.Tensor,
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

    def _match_buffer_rank(self, arg, max_rank):
        """
        Change arg's fake tensor meta to match max_rank if:
            - arg is found in inputs_to_buffers or inputs_to_parameters.
        """
        fake_tensor = get_first_fake_tensor(arg)
        shape = fake_tensor.shape
        rank = len(shape)
        new_shape = list([1] * (max_rank - rank) + list(shape))

        buffer_name = None
        if arg.name in self.exported_program.graph_signature.inputs_to_buffers:
            buffer_name = self.exported_program.graph_signature.inputs_to_buffers[
                arg.name
            ]
        elif arg.name in self.exported_program.graph_signature.inputs_to_parameters:
            buffer_name = self.exported_program.graph_signature.inputs_to_parameters[
                arg.name
            ]
        if buffer_name:
            new_tensor = self.exported_program.state_dict[buffer_name].reshape(
                new_shape
            )
            self.exported_program.state_dict[buffer_name] = new_tensor
            arg.meta["val"] = fake_tensor.fake_mode.from_tensor(
                new_tensor, static_shapes=True
            )

    def call(self, graph_module: GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            node = cast(Node, node)

            if node.op != "call_function" or node.target not in self.targeted_ops:
                continue

            # Calculate max rank of all inputs to node
            max_rank = 1
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

                # If the argument is call_function, match shape by inserting view node.
                if arg.op == "call_function":
                    self._match_op_rank(graph_module, node, arg, max_rank)
                else:
                    # If the argument is a buffer or parameter, adjust shape by changing the fake tensor meta.
                    self._match_buffer_rank(arg, max_rank)

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
