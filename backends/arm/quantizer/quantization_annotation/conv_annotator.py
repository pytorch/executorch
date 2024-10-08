# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.f

# pyre-unsafe

from typing import Callable, List, Optional

import torch
from executorch.backends.arm.quantizer import arm_quantizer_utils
from executorch.backends.arm.quantizer.quantization_annotation import register_annotator
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.ao.quantization.quantizer import QuantizationAnnotation

from torch.fx import Node


@register_annotator("conv")
def _annotate_conv(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
        ]:
            continue
        conv_node = n

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = quantization_config.get_input_act_qspec()

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = quantization_config.get_weight_qspec()

        # adding weight node to the partition as well
        partition_nodes = [conv_node, conv_node.args[1]]

        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = quantization_config.get_bias_qspec()
            partition_nodes.append(bias)

        if arm_quantizer_utils.are_annotated(partition_nodes):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition_nodes):
            continue

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.get_output_act_qspec(),
            _annotated=True,
        )
        arm_quantizer_utils.mark_nodes_as_annotated(partition_nodes)
        annotated_partitions.append(partition_nodes)
    return annotated_partitions
