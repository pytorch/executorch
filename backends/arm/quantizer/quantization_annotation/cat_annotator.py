# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import itertools
from typing import Callable, cast, List, Optional

import torch.fx
from executorch.backends.arm.quantizer import arm_quantizer_utils
from executorch.backends.arm.quantizer.quantization_annotation import register_annotator
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


@register_annotator("cat")
def _annotate_cat(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    cat_partitions = get_source_partitions(gm.graph, [torch.cat], filter_fn)
    cat_partitions = list(itertools.chain.from_iterable(cat_partitions.values()))
    annotated_partitions = []
    for cat_partition in cat_partitions:
        annotated_partitions.append(cat_partition.nodes)
        cat_node = cat_partition.output_nodes[0]
        if arm_quantizer_utils.is_annotated(cat_node):
            continue

        input_acts = cast(list[torch.fx.Node], cat_node.args[0])
        input_act0 = input_acts[0]

        input_act_qspec = quantization_config.get_input_act_qspec()
        shared_with_input0_qspec = SharedQuantizationSpec((input_act0, cat_node))

        input_qspec_map = {}

        # First input is set to input qspec from the quantization config.
        if isinstance(input_act0, Node):
            if not arm_quantizer_utils.is_input_ok_for_quantization(input_act0, gm):
                continue
            input_qspec_map[input_act0] = input_act_qspec

        # For the rest of the inputs, share qspec with first.
        # If we can't quantize any of the inputs, abort annotation.
        for input_act in input_acts[1:]:
            if isinstance(input_act, Node):
                if not arm_quantizer_utils.is_input_ok_for_quantization(input_act, gm):
                    continue
                if input_act is not input_act0:
                    input_qspec_map[input_act] = shared_with_input0_qspec

        if input_qspec_map is not None:
            cat_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=shared_with_input0_qspec,
                _annotated=True,
            )
    return annotated_partitions
