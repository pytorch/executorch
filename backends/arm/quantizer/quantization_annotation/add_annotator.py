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
        if arm_quantizer_utils.is_annotated(add_node):
            continue

        input_qspec_map, output_qspec = arm_quantizer_utils.get_shared_qspec(
            add_node, gm, quantization_config
        )
        if input_qspec_map is not None:
            add_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_qspec,
                _annotated=True,
            )
    return annotated_partitions
