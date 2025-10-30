# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.convert_to_clamp import ConvertToClampPass
from executorch.backends.arm._passes.fold_qdq_with_annotated_qparams_pass import (
    FoldAndAnnotateQParamsPass,
)
from executorch.backends.arm._passes.quant_args import QuantArgs
from executorch.backends.arm.constants import Q_OPS
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import Node


class FuseQuantizedActivationPass(ArmPass):
    _passes_required_after: Set[Type[ExportPass]] = {
        ConvertToClampPass,
        FoldAndAnnotateQParamsPass,
        RemoveGetItemPass,
    }

    @staticmethod
    def _is_fuseable_quantized_activation(node: Node):
        """Fuse activations that have a 0 lower bound and quantized with a qmin zero-point"""
        is_fuseable = node.target == exir_ops.edge.aten.relu.default
        if node.target == exir_ops.edge.aten.hardtanh.default:
            min_val = node.args[1]
            is_fuseable = min_val == 0

        is_quantized = len(node.users) == 1 and next(iter(node.users)).target in Q_OPS
        if is_fuseable and is_quantized:
            quant_node = next(iter(node.users))
            quant_args = QuantArgs.from_operator(quant_node.target, quant_node.args)
            zp = quant_args.zp
            qmin = quant_args.qmin
            return zp == qmin
        else:
            return False

    @staticmethod
    def _is_fuseable_input(node: Node):
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

            if not FuseQuantizedActivationPass._is_fuseable_quantized_activation(node):
                continue

            input_node = node.args[0]
            if not FuseQuantizedActivationPass._is_fuseable_input(input_node):
                continue

            node.replace_all_uses_with(input_node)
            graph_module.graph.erase_node(node)
            modified = True

        if modified:
            graph_module.recompile()
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, modified)
