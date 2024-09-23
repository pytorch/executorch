# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import itertools
import operator
from typing import Callable, List, Optional

import torch
from executorch.backends.arm.quantizer import arm_quantizer_utils
from executorch.backends.arm.quantizer.quantization_annotation import register_annotator
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.ao.quantization.quantizer import QuantizationAnnotation
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


@register_annotator("mul")
def _annotate_mul(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    mul_partitions = get_source_partitions(
        gm.graph, ["mul", "mul_", operator.mul, torch.mul, operator.imul], filter_fn
    )
    mul_partitions = list(itertools.chain.from_iterable(mul_partitions.values()))
    annotated_partitions = []
    for mul_partition in mul_partitions:
        annotated_partitions.append(mul_partition.nodes)
        mul_node = mul_partition.output_nodes[0]
        if arm_quantizer_utils.is_annotated(mul_node):
            continue

        input_act_qspec = quantization_config.get_input_act_qspec()
        output_act_qspec = quantization_config.get_output_act_qspec()

        input_qspec_map = {}
        input_act0 = mul_node.args[0]
        if isinstance(input_act0, Node):
            if arm_quantizer_utils.is_input_large_scalar(input_act0, gm):
                continue
            if arm_quantizer_utils.is_input_non_float_tensor(input_act0):
                continue
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = mul_node.args[1]
        if isinstance(input_act1, Node):
            if arm_quantizer_utils.is_input_large_scalar(input_act1, gm):
                continue
            if arm_quantizer_utils.is_input_non_float_tensor(input_act1):
                continue
            input_qspec_map[input_act1] = input_act_qspec

        mul_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions
