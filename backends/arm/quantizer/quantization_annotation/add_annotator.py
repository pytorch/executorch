# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import operator
from typing import Callable, List, Optional

import torch
from executorch.backends.arm.quantizer import arm_quantizer_utils
from executorch.backends.arm.quantizer.quantization_annotation import register_annotator
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


@register_annotator("add")
def _annotate_add(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    add_partitions = get_source_partitions(
        gm.graph, [operator.add, torch.add, operator.iadd], filter_fn
    )
    add_partitions = list(itertools.chain.from_iterable(add_partitions.values()))
    annotated_partitions = []
    for add_partition in add_partitions:
        annotated_partitions.append(add_partition.nodes)
        add_node = add_partition.output_nodes[0]
        if arm_quantizer_utils.is_annotated([add_node]):
            continue

        input_act0 = add_node.args[0]
        input_act_qspec = quantization_config.get_input_act_qspec()
        shared_with_input0_qspec = SharedQuantizationSpec((input_act0, add_node))

        input_qspec_map = {}
        if isinstance(input_act0, Node):
            if arm_quantizer_utils.is_input_large_scalar(input_act0, gm):
                continue
            if arm_quantizer_utils.is_input_non_float_tensor(input_act0):
                continue
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            if arm_quantizer_utils.is_input_large_scalar(input_act1, gm):
                continue
            if arm_quantizer_utils.is_input_non_float_tensor(input_act1):
                continue
            if input_act0 is not input_act1:
                input_qspec_map[input_act1] = shared_with_input0_qspec
            else:
                input_qspec_map[input_act1] = input_act_qspec

        add_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=shared_with_input0_qspec,
            _annotated=True,
        )
    return annotated_partitions
