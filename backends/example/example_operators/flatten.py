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


def _annotate_flatten(partitions, quant_config):
    """
    This is what the graph of a simple add op looks like:
    add_tensor = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    """
    flatten_node = partitions[0].output_nodes[0]
    flatten_input = flatten_node.args[0]

    if _nodes_are_annotated([flatten_node]):
        return

    _annotate_nodes(
        [(flatten_node, flatten_input)], quant_config.input_quant_spec, input_node=True
    )
    _annotate_nodes([(flatten_node,)], quant_config.output_quant_spec)


@dataclass
class FlattenNode(OpBase):
    def __init__(self):
        super().__init__(
            pattern=(torch.flatten,),
            annotate_handle=_annotate_flatten,
        )
