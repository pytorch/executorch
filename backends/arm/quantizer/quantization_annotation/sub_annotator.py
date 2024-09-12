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
from torch.fx import GraphModule, Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


@register_annotator("sub")
def _annotate_sub(
    gm: GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    sub_partitions = get_source_partitions(
        gm.graph, [operator.sub, torch.sub, operator.isub], filter_fn
    )
    sub_partitions = list(itertools.chain.from_iterable(sub_partitions.values()))
    annotated_partitions = []
    for sub_partition in sub_partitions:
        annotated_partitions.append(sub_partition.nodes)
        sub_node = sub_partition.output_nodes[0]
        if arm_quantizer_utils.is_annotated(sub_node):
            continue

        input_qspec_map, output_qspec = arm_quantizer_utils.get_shared_qspec(
            sub_node, gm, quantization_config
        )
        if input_qspec_map is not None:
            sub_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_qspec,
                _annotated=True,
            )
    return annotated_partitions
