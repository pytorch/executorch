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


def _annotate_linear(partitions, quant_config):
    """
    This is what the graph of a simple linear op looks like:
    fn_weight = self.fn_weight
    fn_bias = self.fn_bias
    permute_copy = torch.ops.aten.permute_copy.default(fn_weight, [1, 0]);  fn_weight = None
    addmm = torch.ops.aten.addmm.default(fn_bias, arg2_1, permute_copy);  fn_bias = arg2_1 = permute_copy = None
    """
    linear_node = partitions[0].output_nodes[0]
    if _nodes_are_annotated([linear_node]):
        return

    input_node = linear_node.args[0]
    # permute_node = linear_node.args[1]
    # print("permute_node: ", permute_node, " args: ", permute_node.args, " target: ", permute_node.target)
    weight_node = linear_node.args[1]
    print(
        "weight_node: ",
        weight_node,
        " args: ",
        weight_node.args,
        " target: ",
        weight_node.target,
    )
    # Unused.
    # bias_node = output_node.args[0]

    # if _nodes_are_annotated([linear_node, permute_node]):
    #     return

    _annotate_nodes(
        [(linear_node, input_node)], quant_config.input_quant_spec, input_node=True
    )
    _annotate_nodes(
        [(linear_node, weight_node)], quant_config.weight_quant_spec, input_node=True
    )
    _annotate_nodes([(linear_node,)], quant_config.output_quant_spec)


@dataclass
class LinearNode(OpBase):
    def __init__(self):
        super().__init__(
            pattern=(torch.nn.Linear,),
            annotate_handle=_annotate_linear,
        )
