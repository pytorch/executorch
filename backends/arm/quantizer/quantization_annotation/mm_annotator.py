# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import itertools
from typing import Callable, List, Optional

import torch
from executorch.backends.arm.quantizer import arm_quantizer_utils
from executorch.backends.arm.quantizer.quantization_annotation import register_annotator
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.ao.quantization.quantizer import QuantizationAnnotation
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


@register_annotator("mm")
def _annotate_mm(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    mm_partitions = get_source_partitions(gm.graph, [torch.mm, torch.bmm], filter_fn)
    mm_partitions = list(itertools.chain.from_iterable(mm_partitions.values()))
    annotated_partitions = []
    for mm_partition in mm_partitions:
        annotated_partitions.append(mm_partition.nodes)
        mm_node = mm_partition.output_nodes[0]

        if arm_quantizer_utils.is_annotated(mm_node):
            continue

        input_act_qspec = quantization_config.get_input_act_qspec()
        output_act_qspec = quantization_config.get_output_act_qspec()

        input_qspec_map = {}
        input_act0 = mm_node.args[0]
        if isinstance(input_act0, Node):
            if not arm_quantizer_utils.is_input_ok_for_quantization(input_act0, gm):
                continue
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = mm_node.args[1]
        if isinstance(input_act1, Node):
            if not arm_quantizer_utils.is_input_ok_for_quantization(input_act1, gm):
                continue
            input_qspec_map[input_act1] = input_act_qspec

        mm_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions
