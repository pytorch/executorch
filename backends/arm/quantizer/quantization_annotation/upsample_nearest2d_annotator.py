# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import itertools
from typing import Callable, List, Optional

import torch
from executorch.backends.arm.quantizer.quantization_annotation import register_annotator
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


def _filter_upsample_nearest2d(filter_fn: Optional[Callable[[Node], bool]] = None):
    def filter(node: Node):
        is_upsample = node.target == torch.ops.aten.upsample_nearest2d.vec
        if filter_fn is None:
            return is_upsample
        else:
            return is_upsample and filter_fn(node)

    return filter


@register_annotator("upsample_nearest2d")
def _annotate_upsample_nearest2d(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    module_partitions = get_source_partitions(
        gm.graph,
        [
            torch.nn.UpsamplingNearest2d,
            torch.nn.Upsample,
            torch.nn.functional.interpolate,
        ],
        _filter_upsample_nearest2d(filter_fn),
    )
    upsample_partitions = list(
        itertools.chain.from_iterable(module_partitions.values())
    )
    annotated_partitions = []

    for upsample_partition in upsample_partitions:
        annotated_partitions.append(upsample_partition.nodes)

        assert len(upsample_partition.nodes) == 1
        upsample_node = upsample_partition.nodes[0]

        input_act = upsample_node.args[0]
        assert isinstance(input_act, Node)

        input_act_qspec = quantization_config.get_input_act_qspec()
        output_act_qspec = SharedQuantizationSpec((input_act, upsample_node))

        upsample_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                input_act: input_act_qspec,
            },
            output_qspec=output_act_qspec,
            _annotated=True,
        )

    return annotated_partitions
