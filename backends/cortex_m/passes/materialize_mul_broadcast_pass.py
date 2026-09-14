# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.cortex_m.passes.passes_utils import (
    is_channel_broadcast,
    is_channels_last,
)
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule, Node


class MaterializeMulBroadcastPass(ExportPass):
    """Materialize MUL broadcasts unsupported by the Cortex-M kernel.

    ``cortex_m.quantized_mul`` supports equal-sized tensors and the backend's
    channels-last channel-broadcast form. Other valid PyTorch broadcasting
    forms are expanded before quantization so that the resulting MUL can be
    annotated and lowered to the Cortex-M kernel.
    """

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False

        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target != torch.ops.aten.mul.Tensor:
                continue

            lhs, rhs = node.args[:2]

            if not isinstance(lhs, Node) or not isinstance(rhs, Node):
                continue

            lhs_tensor = get_first_fake_tensor(lhs)
            rhs_tensor = get_first_fake_tensor(rhs)
            output_tensor = get_first_fake_tensor(node)

            if lhs_tensor.shape == rhs_tensor.shape:
                continue

            if is_channel_broadcast(lhs_tensor, rhs_tensor) and is_channels_last(
                lhs_tensor
            ):
                continue

            output_shape = list(output_tensor.shape)
            new_args = list(node.args)

            for index, arg in enumerate((lhs, rhs)):
                tensor = get_first_fake_tensor(arg)

                if tensor.shape == output_tensor.shape:
                    continue

                with graph_module.graph.inserting_before(node):
                    expanded = graph_module.graph.call_function(
                        torch.ops.aten.expand_copy.default,
                        args=(arg, output_shape),
                    )

                new_args[index] = expanded
                modified = True

            node.args = tuple(new_args)

        if modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
