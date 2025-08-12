# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.backends.arm._passes.arm_pass_utils import (
    create_node,
    get_first_fake_tensor,
    insert_q_dq_pair,
)
from executorch.backends.arm.constants import DQ_OPS, Q_OPS
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import Node


class ConvertMmToBmmPass(ExportPass):
    """
    This pass converts a MM node to a BMM one and turns input and output tensors
    from rank 2 to rank 3. The TOSA specification requires rank 3. The graph is
    modified to do the following:
    1) Unsqueeze input tensors to rank 3.
    2) Convert MM node to BMM.
    3) Squeeze output tensor to rank 2.
    """

    def call(self, graph_module: torch.fx.GraphModule):
        modified_graph = False
        graph = graph_module.graph
        node_list = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.mm.default
        )
        for node in node_list:
            # Unsqueeze input tensors to rank 3
            for input_node in node.args:
                if not isinstance(input_node, Node):
                    continue

                shape = get_first_fake_tensor(input_node).shape
                rank = len(shape)
                if rank != 2:
                    raise RuntimeError(f"Input tensor has rank {rank}, must be 2")

                with graph.inserting_before(node):
                    unsqueeze_before = create_node(
                        graph, exir_ops.edge.aten.unsqueeze_copy.default, from_node=node
                    )
                    unsqueeze_before.args = (
                        input_node,  # Input is node's original input
                        0,
                    )
                    node.replace_input_with(input_node, unsqueeze_before)

                # If Quantized we must insert unsqueeze --> q --> dq --> node
                if input_node.target in DQ_OPS:
                    q_params = input_node.args[1:]
                    insert_q_dq_pair(graph, unsqueeze_before, q_params, from_node=node)

            # Replace mm node with bmm
            with graph.inserting_before(node):
                bmm_node = create_node(
                    graph,
                    exir_ops.edge.aten.bmm.default,
                    from_node=node,
                )
                bmm_node.args = node.args
                node.replace_all_uses_with(bmm_node)
                graph.erase_node(node)

            # Unsqueeze output tensor to rank 3
            with graph.inserting_after(bmm_node):
                squeeze_after = create_node(
                    graph,
                    exir_ops.edge.aten.squeeze_copy.dims,
                    from_node=node,
                )
                squeeze_after.args = (
                    bmm_node,
                    [0],
                )
                original_users = [
                    user for user in bmm_node.users if user != squeeze_after
                ]
                for user in original_users:
                    user.replace_input_with(bmm_node, squeeze_after)

            # If quantized, insert mm --> q --> dq --> squeeze
            if all(original_user.target in Q_OPS for original_user in original_users):
                q_params = original_users[0].args[1:]
                insert_q_dq_pair(graph, bmm_node, q_params, from_node=node)

            modified_graph = True

        if modified_graph:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified_graph)
