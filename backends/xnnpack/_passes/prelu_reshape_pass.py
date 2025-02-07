# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.utils import (
    check_or_raise,
    get_param_tensor,
    is_param_node,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import PassResult


class PReLUReshapePass(XNNPACKPass):
    """
    This pass is used to modify the args of a PReLU node to make it compatible
    with running via XNNPACK delegate. If there is only one parameter in the
    weight tensor, repeat it to make the tensor to length num_channels.
    This is because pytorch supports having either per-tensor or per-channel
    weight parameters for PReLU, whereas XNNPACK supports only per-channel
    """

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        node_list = list(graph.nodes)
        for node in node_list:
            if node.op == "call_function":
                if node.target == exir_ops.edge.aten._prelu_kernel.default:
                    weight_node = node.args[1]

                    check_or_raise(
                        is_param_node(self.exported_program, weight_node),
                        "Only constant weight PReLU is supported by XNNPACK",
                    )

                    weight_data = get_param_tensor(self.exported_program, weight_node)
                    if weight_data is None:
                        raise AssertionError("Expected weight tensor to be not None")

                    weight_data = weight_data.data.contiguous()

                    check_or_raise(
                        weight_data.dim() == 4,
                        f"4D weight required for XNNPACK PReLU, got: {weight_data.dim()}D",
                    )

                    if weight_data.numel() == 1:
                        input_shape = node.args[0].meta["val"].shape

                        check_or_raise(
                            len(input_shape) == 4,
                            f"4D input required for XNNPACK PReLU, got: {len(input_shape)}D",
                        )

                        num_channels = input_shape[1]

                        weight_data_per_channel = weight_data.repeat(
                            1, num_channels, 1, 1
                        )

                        setattr(
                            weight_node.graph.owning_module,
                            weight_node.target,
                            torch.nn.Parameter(data=weight_data_per_channel),
                        )

        # Since we are overriding "call", we need to call the parent's "call"
        # to retrace the graph and regenerate metadata
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
