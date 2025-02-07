# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from executorch.backends.arm.tosa_quant_utils import q_op
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import Node


class FuseQuantizedActivationPass(ExportPass):
    def _is_fuseable_quantized_activation(self, node: Node):
        """Fuse activations that have a 0 lower bound and quantized with a qmin zero-point"""
        is_fuseable = node.target == exir_ops.edge.aten.relu.default
        if node.target == exir_ops.edge.aten.hardtanh.default:
            min_val = node.args[1]
            is_fuseable = min_val == 0

        is_quantized = len(node.users) == 1 and next(iter(node.users)).target == q_op
        if is_fuseable and is_quantized:
            quant_node = next(iter(node.users))
            zp = quant_node.args[2]
            qmin = quant_node.args[3]
            return zp == qmin
        else:
            return False

    def _is_fuseable_input(self, node: Node):
        return (
            node.target
            in (
                exir_ops.edge.aten.convolution.default,
                exir_ops.edge.aten.linear.default,
            )
            and len(node.users) == 1
        )

    def call(self, graph_module: torch.fx.GraphModule):
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue

            if not self._is_fuseable_quantized_activation(node):
                continue

            input_node = node.args[0]
            if not self._is_fuseable_input(input_node):
                continue

            node.replace_all_uses_with(input_node)
            graph_module.graph.erase_node(node)
            modified = True

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
