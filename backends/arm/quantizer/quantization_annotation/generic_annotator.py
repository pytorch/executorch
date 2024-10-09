# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Callable, List, Optional

import torch
import torch.fx
from executorch.backends.arm.quantizer import arm_quantizer_utils
from executorch.backends.arm.quantizer.quantization_annotation import register_annotator
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torch.ao.quantization.quantizer import SharedQuantizationSpec
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)
from torch.fx import Node


_SUPPORTED_OPS = [
    # DATA LAYOUT OPS
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze_copy.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.unsqueeze_copy.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.repeat.default,
    torch.ops.aten.expand_copy.default,
    torch.ops.aten.expand.default,
    # Disabling these as there seems to be an issue with support for complex
    # datatypes in torch:
    # torch.ops.aten.view_as_complex.default,
    # torch.ops.aten.view_as_complex_copy.default,
    # torch.ops.aten.view_as_real.default,
    # torch.ops.aten.view_as_real_copy.default,
    torch.ops.aten.view_copy.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.slice_copy.Tensor,
    # 'concat' should be handled separately as it has a sequence of inputs and
    # makes the implementation unnecessary complicated.
    # torch.ops.aten.concat.default,
    torch.ops.aten.transpose.Dimname,
    torch.ops.aten.transpose.int,
    torch.ops.aten.transpose_copy.int,
    torch.ops.aten.tile.default,
    torch.ops.aten.flip.default,
]


@register_annotator("generic")
def _annotate_generic(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """Propagate qspecs to generic ops like unsqueeze, reshape etc."""
    annotated_partitions = []

    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in _SUPPORTED_OPS:
            continue
        if filter_fn and not filter_fn(node):
            continue
        if arm_quantizer_utils.is_annotated(node):
            continue

        input_node = node.args[0]

        # Using a non-shared quantization spec here as a SharedQuantizationSpec
        # can lead to a recursion.
        _annotate_input_qspec_map(
            node, input_node, quantization_config.get_input_act_qspec()
        )
        _annotate_output_qspec(node, SharedQuantizationSpec((input_node, node)))

        arm_quantizer_utils.mark_nodes_as_annotated([node])
        annotated_partitions.append([node])

    return annotated_partitions
