# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from executorch.backends.transforms import get_shape
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.quant_utils import (
    insert_q_dq_pair,
    is_dequant,
    is_quant,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult
from torch._ops import OpOverload


class MeanDimRewritePass(XNNPACKPass):
    """
    This pass rewrites mean operations that reduce only the last dimension with keepdim=True
    into a form that XNNPACK can handle. Specifically, it transforms:

    x.mean(-1, keepdim=True)

    into:

    x.unsqueeze(-1).mean([-2, -1], keepdim=True).squeeze(-1)

    This allows XNNPACK to handle single dimension mean operations by converting them
    to the supported 2D mean operation pattern. Note: XNNPACK only supports unsqueezing
    in the last dimension, so we add the new dimension at the end.
    """

    def create_node(
        self,
        graph: torch.fx.Graph,
        op_target: OpOverload,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ):
        return graph.create_node(
            "call_function",
            op_target,
            args=args,
            kwargs=kwargs or {},
        )

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        node_list = list(graph.nodes)

        for node in node_list:
            if node.op == "call_function" and node.target.__name__ == "aten.mean.dim":
                # Check if this is a mean operation on the last dimension with keepdim=True
                if len(node.args) >= 3:
                    dims = node.args[1]
                    keepdim = node.args[2]

                    # Only transform if dims is [-1] and keepdim is True
                    if dims == [-1] and keepdim is True:
                        # Get the input tensor
                        input_node = node.args[0]

                        # Check if input is 3D tensor (will become 4D after unsqueeze)
                        input_shape = get_shape(input_node)
                        if len(input_shape) == 3:
                            # Insert unsqueeze before the mean operation
                            with graph.inserting_before(node):
                                unsqueeze_node = self.create_node(
                                    graph,
                                    exir_ops.edge.aten.unsqueeze_copy.default,
                                    args=(input_node, -1),
                                )
                                # Update the unsqueeze node's metadata
                                unsqueeze_shape = list(input_shape)
                                unsqueeze_shape.append(
                                    1
                                )  # Append 1 to the last dimension
                                unsqueeze_node.meta = input_node.meta.copy()
                                unsqueeze_node.meta["val"] = torch.empty(
                                    unsqueeze_shape, dtype=input_node.meta["val"].dtype
                                )

                                # Update the mean operation to use unsqueeze output and reduce both dims
                                node.args = (unsqueeze_node, [-2, -1], keepdim)

                            # Handle quantization for unsqueeze: if input is quantized, insert q->dq after unsqueeze
                            if is_dequant(input_node):
                                q_params = input_node.args[1:]
                                insert_q_dq_pair(graph, unsqueeze_node, q_params)

                            # Insert squeeze after the mean operation
                            with graph.inserting_after(node):
                                squeeze_node = self.create_node(
                                    graph,
                                    exir_ops.edge.aten.squeeze_copy.dim,
                                    args=(node, -1),
                                )
                                # Update squeeze node's metadata - should match the original mean output
                                squeeze_node.meta = node.meta.copy()

                            # Replace all uses of the original mean node with the squeeze node
                            original_users = [
                                user for user in node.users if user != squeeze_node
                            ]
                            for user in original_users:
                                user.replace_input_with(node, squeeze_node)

                            # Handle quantization for mean->squeeze: if original users are quantized, insert q->dq after mean
                            if all(
                                is_quant(original_user)
                                for original_user in original_users
                            ):
                                q_params = original_users[0].args[1:]
                                insert_q_dq_pair(graph, node, q_params)

        graph_module.recompile()
        # Since we are overriding "call", we need to call the parent's "call"
        # to retrace the graph and regenerate metadata
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
