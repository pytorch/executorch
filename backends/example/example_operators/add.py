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


def _annotate_add(partitions, quant_config):
    """
    This is what the graph of a simple add op looks like:
    add_tensor = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    """
    add_node = partitions[0].output_nodes[0]
    add_input_1 = add_node.args[0]
    add_input_2 = add_node.args[1]

    if _nodes_are_annotated([add_node]):
        return

    _annotate_nodes(
        [(add_node, add_input_1)], quant_config.input_quant_spec, input_node=True
    )
    _annotate_nodes(
        [(add_node, add_input_2)], quant_config.weight_quant_spec, input_node=True
    )
    _annotate_nodes([(add_node,)], quant_config.output_quant_spec)


@dataclass
class AddNode(OpBase):
    def __init__(self):
        super().__init__(
            pattern=(torch.add,),
            annotate_handle=_annotate_add,
        )
