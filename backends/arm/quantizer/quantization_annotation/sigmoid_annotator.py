# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Callable, List, Optional

import torch
from executorch.backends.arm.quantizer import arm_quantizer_utils
from executorch.backends.arm.quantizer.quantization_annotation import register_annotator
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)
from torch.fx import Node


@register_annotator("sigmoid")
def _annotate_sigmoid(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []

    # input/ output range of sigmoid is always same -> quantize with fixed qspec.
    # this configuration maps input: (-128, 127) -> (-6.0, 5.95). Outside these bounds, sigmoid ~= const.
    #                        output: (-1,0.99) -> (-128, 127). Sigmoid has output value range (-1,1)
    # Note that this exact choice is somewhat arbitrary.

    input_act_qspec = quantization_config.get_fixed_qspec(scale=6 / 128, zp=0)
    output_act_qspec = quantization_config.get_fixed_qspec(scale=1 / 128, zp=0)

    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.sigmoid.default:
            continue
        if filter_fn and not filter_fn(node):
            continue
        input_node = node.args[0]

        if not arm_quantizer_utils.is_annotated(node):
            _annotate_input_qspec_map(
                node,
                input_node,
                input_act_qspec,
            )
            _annotate_output_qspec(node, output_act_qspec)

            arm_quantizer_utils.mark_nodes_as_annotated([node])
            annotated_partitions.append([node])

    return annotated_partitions
