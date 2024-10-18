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
from torch.ao.quantization.quantizer import QuantizationAnnotation
from torch.fx import GraphModule, Node


@register_annotator("sub")
def _annotate_sub(
    gm: GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for node in gm.graph.nodes:
        if node.target not in (torch.ops.aten.sub.Tensor, torch.ops.aten.sub_.Tensor):
            continue
        annotated_partitions.append(node)
        sub_node = node
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
