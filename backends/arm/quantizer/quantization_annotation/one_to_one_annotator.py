# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Callable, List, Optional

import torch
import torch.fx
from executorch.backends.arm.quantizer import arm_quantizer_utils
from executorch.backends.arm.quantizer.quantization_annotation import register_annotator
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)
from torch.fx import Node


@register_annotator("one_to_one")
def _annotate_one_to_one(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """
    This annotator adds the input and output qspec from the quantization config to
    ops in 'one_to_one_ops' that have the following properties:
    - Have a single input and single output.
    - Can handle different qspecs on the input and output.

    Typical ops are ops implemented with a lookup table.
    """
    annotated_partitions = []
    one_to_one_ops = (torch.ops.aten.exp.default, torch.ops.aten.log.default)
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in one_to_one_ops:
            continue
        if filter_fn and not filter_fn(node):
            continue
        input_node = node.args[0]

        if not arm_quantizer_utils.is_annotated(node):
            _annotate_input_qspec_map(
                node,
                input_node,
                quantization_config.get_input_act_qspec(),
            )
            _annotate_output_qspec(node, quantization_config.get_output_act_qspec())

            arm_quantizer_utils.mark_nodes_as_annotated([node])
            annotated_partitions.append([node])

    return annotated_partitions
