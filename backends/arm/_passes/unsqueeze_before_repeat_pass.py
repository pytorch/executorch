# Copyright 2024-2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe
from typing import Set, Type

import torch
import torch.fx
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class UnsqueezeBeforeRepeatPass(ArmPass):
    """
    A TOSA TILE op only supports rank(in) == rank(out).
    To support Pytorch's repeat which can also add dimensions,
    we add an explicit view op before which adds the new dimensions.
    New dimensions are appendend at the front, see
    https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html

    Original:
        repeat(multiples)
    After pass:
        view(shape = [1]*num_new_dims + old_shape)
        repeat(multiples)
    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    def call(self, graph_module: torch.fx.GraphModule):
        modified_graph = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target != exir_ops.edge.aten.repeat.default:
                continue

            old_shape = list(get_first_fake_tensor(node.all_input_nodes[0]).shape)
            old_rank = len(old_shape)
            multiples = node.args[1]
            new_rank = len(multiples)
            if old_rank == new_rank:
                continue

            num_new_dims = new_rank - old_rank
            new_shape = [1] * num_new_dims + old_shape

            with graph_module.graph.inserting_before(node):
                view_node = create_node(
                    graph_module.graph,
                    exir_ops.edge.aten.view_copy.default,
                    (node.all_input_nodes[0], new_shape),
                )
                node.replace_input_with(node.all_input_nodes[0], view_node)
            modified_graph = True

        if modified_graph:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, modified_graph)
