# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import executorch.backends.vulkan.utils as utils
import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

#############################
## aten.weight_int8pack_mm ##
#############################


def matches_int8pack_mm_pattern(node: torch.fx.Node) -> bool:
    if not utils.is_linear_node(node):
        return False

    input_node = node.args[0]
    weight_node = node.args[1]

    # Type checking
    if not isinstance(weight_node, torch.fx.Node):
        return False
    if not isinstance(input_node, torch.fx.Node):
        return False

    # The weight arg should be a dequant node dequantizing the quantized weight
    # Furthermore, the op expects per channel quantization of the weight
    if not utils.is_dequant_per_channel_node(weight_node):
        return False

    orig_weight = weight_node.args[0]
    if not isinstance(orig_weight, torch.fx.Node):
        return False

    # The quantized weight data should be a int8 tensor
    if orig_weight.meta["val"].dtype != torch.int8:
        return False

    # The input arg should not be a dequant node
    if utils.is_dequant_node(input_node):
        return False

    return True


def fuse_into_weight_int8pack_mm_node(
    graph_module: torch.fx.GraphModule,
    linear_node: torch.fx.Node,
) -> None:
    """
    The weight_int8pack_mm operator represents a weight only quantized linear operator.
    After the PT2E quantization flow, the expected graph pattern is

        dq_weight = dequantize(weight, scales)
        out = linear(activation, dq_weight, bias?)

    The goal of this function is to condense that sequence into

        out = weight_int8pack_mm(activation, dq_weight, scales)
        out = out + bias
    """
    activation = linear_node.args[0]
    dq_weight_node = linear_node.args[1]
    assert isinstance(activation, torch.fx.Node)
    assert isinstance(dq_weight_node, torch.fx.Node)

    bias = None
    if len(linear_node.args) > 2:
        bias = linear_node.args[2]
        assert isinstance(bias, torch.fx.Node)

    orig_weight = dq_weight_node.args[0]
    scale = dq_weight_node.args[1]

    with graph_module.graph.inserting_before(linear_node):
        weight_int8pack_mm_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.aten._weight_int8pack_mm.default,
            (activation, orig_weight, scale),
        )
        if bias:
            add_node = graph_module.graph.create_node(
                "call_function",
                exir_ops.edge.aten.add.Tensor,
                (weight_int8pack_mm_node, bias),
            )
            linear_node.replace_all_uses_with(add_node)
        else:
            linear_node.replace_all_uses_with(weight_int8pack_mm_node)
        graph_module.graph.erase_node(linear_node)
        graph_module.graph.erase_node(dq_weight_node)


class FuseQuantizedOpsTransform(ExportPass):
    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if matches_int8pack_mm_pattern(node):
                fuse_into_weight_int8pack_mm_node(graph_module, node)

        graph_module.recompile()
        graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, True)
