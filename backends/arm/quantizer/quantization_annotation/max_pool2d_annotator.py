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
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)
from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


@register_annotator("max_pool2d")
def _annotate_max_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.MaxPool2d, torch.nn.functional.max_pool2d], filter_fn
    )
    maxpool_partitions = list(itertools.chain.from_iterable(module_partitions.values()))
    annotated_partitions = []
    for maxpool_partition in maxpool_partitions:
        annotated_partitions.append(maxpool_partition.nodes)
        output_node = maxpool_partition.output_nodes[0]
        maxpool_node = None
        for n in maxpool_partition.nodes:
            if n.target == torch.ops.aten.max_pool2d.default:
                maxpool_node = n
        assert (
            maxpool_node is not None
        ), "ArmQuantizer only works with torch.ops.aten.max_pool2d.default, "
        "please make sure you are exporting the model correctly"
        if arm_quantizer_utils.are_annotated([output_node, maxpool_node]):  # type: ignore[list-item]
            continue

        input_act = maxpool_node.args[0]  # type: ignore[union-attr]
        assert isinstance(input_act, Node)

        # only annotate maxpool when the output of the input node is annotated
        if (
            "quantization_annotation" not in input_act.meta
            or not input_act.meta["quantization_annotation"]._annotated
            or input_act.meta["quantization_annotation"].output_qspec is None
        ):
            continue
        # input and output of maxpool will share quantization parameter with input of maxpool
        act_qspec = SharedQuantizationSpec(input_act)
        # act_qspec = get_act_qspec(quantization_config)
        maxpool_node.meta["quantization_annotation"] = QuantizationAnnotation(  # type: ignore[union-attr]
            input_qspec_map={
                input_act: act_qspec,
            },
            _annotated=True,
        )
        output_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=act_qspec,
            _annotated=True,
        )
    return annotated_partitions
