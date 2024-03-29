# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from executorch.backends.example.example_operators.op_base import OpBase
from executorch.backends.example.example_operators.utils import (
    _annotate_nodes,
    _nodes_are_annotated,
)


def _annotate_conv_relu(partitions, quant_config):
    """
    This is what the graph of a simple conv + relu pattern looks like:
    l__self___conv_weight = self.L__self___conv_weight
    l__self___conv_bias = self.L__self___conv_bias
    convolution_default = torch.ops.aten.convolution.default(arg2_1, l__self___conv_weight, l__self___conv_bias, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg2_1 = l__self___conv_weight = l__self___conv_bias = None
    relu_default = torch.ops.aten.relu.default(convolution_default);  convolution_default = None
    """

    conv_node = partitions[0].output_nodes[0]
    input_node = conv_node.args[0]
    relu_node = partitions[1].output_nodes[0]
    weight_node = conv_node.args[1]

    if _nodes_are_annotated([conv_node, relu_node]):
        return

    _annotate_nodes(
        [(conv_node, input_node)], quant_config.input_quant_spec, input_node=True
    )
    _annotate_nodes(
        [(conv_node, weight_node)], quant_config.weight_quant_spec, input_node=True
    )
    _annotate_nodes([(relu_node,)], quant_config.output_quant_spec)


@dataclass
class ConvReluNode(OpBase):
    def __init__(self):
        super().__init__(
            pattern=(torch.nn.Conv2d, torch.nn.ReLU),
            annotate_handle=_annotate_conv_relu,
        )
