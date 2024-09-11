# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import itertools
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from executorch.backends.arm.quantizer import arm_quantizer_utils
from executorch.backends.arm.quantizer.quantization_annotation import register_annotator
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


@register_annotator("adaptive_avg_pool2d")
def _annotate_adaptive_avg_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """Always annotate adaptive_avg_pool2d op"""
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d], filter_fn
    )
    partitions = list(itertools.chain.from_iterable(module_partitions.values()))
    annotated_partitions = []
    for partition in partitions:
        pool_node = partition.output_nodes[0]
        if (
            pool_node.op != "call_function"
            or pool_node.target != torch.ops.aten.adaptive_avg_pool2d.default
        ):
            raise ValueError(f"{pool_node} is not an aten adaptive_avg_pool2d operator")

        if arm_quantizer_utils.is_annotated(pool_node):
            continue

        annotated_partitions.append(partition.nodes)
        input_act = pool_node.args[0]
        assert isinstance(input_act, Node)

        # only annotate input output sharing operator
        # when the output of the input node is annotated
        if (
            "quantization_annotation" not in input_act.meta
            or not input_act.meta["quantization_annotation"]._annotated
            or input_act.meta["quantization_annotation"].output_qspec is None
        ):
            input_act_qspec = quantization_config.get_input_act_qspec()
        else:
            input_act_qspec = SharedQuantizationSpec(input_act)

        # output sharing with input
        output_act_qspec = SharedQuantizationSpec((input_act, pool_node))
        pool_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                input_act: input_act_qspec,
            },
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions
