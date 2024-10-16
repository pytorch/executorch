# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, cast, List, Optional

import torch
from executorch.backends.arm.quantizer import arm_quantizer_utils
from executorch.backends.arm.quantizer.quantization_annotation import register_annotator
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig

from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpecBase,
    SharedQuantizationSpec,
)
from torch.fx import Node


@register_annotator("sum")
def _annotate_sum(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for node in gm.graph.nodes:
        if node.target is not torch.ops.aten.sum.dim_IntList:
            continue
        if filter_fn and not filter_fn(node):
            continue

        sum_node = node
        if arm_quantizer_utils.is_annotated(sum_node):
            continue

        input_act = sum_node.args[0]

        if not isinstance(input_act, Node):
            continue
        if not arm_quantizer_utils.is_input_ok_for_quantization(input_act, gm):
            continue

        input_act_qspec = cast(
            Optional[QuantizationSpecBase], quantization_config.get_input_act_qspec()
        )
        input_qspec_map = {input_act: input_act_qspec}
        shared_with_input0_qspec = SharedQuantizationSpec((input_act, sum_node))

        sum_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=shared_with_input0_qspec,
            _annotated=True,
        )
        annotated_partitions.append([sum_node])
    return annotated_partitions
