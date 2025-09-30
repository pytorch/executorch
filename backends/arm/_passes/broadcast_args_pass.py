# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes import ArmPass

from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)

from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class BroadcastArgsPass(ArmPass):
    """
    Pass to manually broadcast arguments by inserting repeats.
    This is done when more than one arg needs broadcasting.
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    targeted_ops = {
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Tensor,
        # mul is indirectly targeting div as div is decompsed to reciprocal + mul
        exir_ops.edge.aten.mul.Tensor,
    }

    def call(self, graph_module: GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.op != "call_function" or node.target not in self.targeted_ops:
                continue

            output_shape = get_first_fake_tensor(node).shape
            nbr_of_broacasts = 0
            for arg in node.args:
                if not isinstance(arg, Node):
                    continue

                shape = get_first_fake_tensor(arg).shape
                if shape != output_shape:
                    nbr_of_broacasts += 1
                if nbr_of_broacasts > 1:
                    multiples = [
                        int(output_shape[d] / shape[d])
                        for d in range(len(output_shape))
                    ]
                    with graph_module.graph.inserting_before(node):
                        repeat = create_node(
                            graph_module.graph,
                            exir_ops.edge.aten.repeat.default,
                            args=(arg, multiples),
                            kwargs={},
                            from_node=node,
                        )
                        node.replace_input_with(arg, repeat)

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
