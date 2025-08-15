# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Callable, List, Optional

import executorch.backends.vulkan.patterns as vk_patterns
import executorch.backends.vulkan.utils as utils

import torch
import torch.nn.functional as F

from executorch.backends.transforms.utils import get_param_tensor, is_param_node

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult

from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher


def fuse_pattern(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    patterns: List[torch.fx.GraphModule],
    create_replacement_func: Callable,
) -> int:
    total_replaced = 0

    for pattern in patterns:
        sm = SubgraphMatcher(pattern.graph, ignore_literals=True)
        matches = list(sm.match(graph_module.graph))

        for partition_to_replace in matches:
            create_replacement_func(ep, graph_module, partition_to_replace)
            total_replaced += 1
            # Remove dead code so they won't be matched again
            graph_module.graph.eliminate_dead_code()

    return total_replaced


##
## Rotary Embedding
##


def identify_rotary_emb_io_nodes(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: InternalMatch,
) -> Optional[List[torch.fx.Node]]:
    # Get the input placeholders (xq, xk, freqs_cos, freqs_sin)
    placeholder_nodes = match.placeholder_nodes
    if len(placeholder_nodes) != 4:
        return None

    xq, xk, freqs_cos, freqs_sin = placeholder_nodes

    output_nodes = match.returning_nodes
    if len(output_nodes) != 2:
        return None

    xq_out, xk_out = output_nodes

    return [xq, xk, freqs_cos, freqs_sin, xq_out, xk_out]


def create_rotary_emb_custom_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: InternalMatch,
):
    io_nodes = identify_rotary_emb_io_nodes(ep, graph_module, match)
    if io_nodes is None:
        return

    assert len(io_nodes) == 6
    xq, xk, freqs_cos, freqs_sin, xq_out, xk_out = io_nodes

    # Create the custom op node
    with graph_module.graph.inserting_before(xq_out):
        rotary_emb_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.apply_rotary_emb.default,
            args=(xq, xk, freqs_cos, freqs_sin),
        )

    # The custom op returns a tuple (xq_out, xk_out)
    # We need to extract the individual outputs
    with graph_module.graph.inserting_after(rotary_emb_node):
        getitem_0 = graph_module.graph.create_node(
            "call_function",
            operator.getitem,
            args=(rotary_emb_node, 0),
        )
        getitem_1 = graph_module.graph.create_node(
            "call_function",
            operator.getitem,
            args=(rotary_emb_node, 1),
        )

    if hasattr(xq_out, "meta") and "val" in xq_out.meta:
        getitem_0.meta["val"] = xq_out.meta["val"]
    if hasattr(xk_out, "meta") and "val" in xk_out.meta:
        getitem_1.meta["val"] = xk_out.meta["val"]

    xq_out.replace_all_uses_with(getitem_0)
    xk_out.replace_all_uses_with(getitem_1)


##
## Quantized Linear
##


def pack_4bit_weight_tensor(inp: torch.Tensor) -> torch.Tensor:
    """
    Given a 8-bit weight tensor containing values quantized to 4 bits, create a packed
    weight tensor by packing 2 4-bit values in one unsigned 8-bit value.

    An input weight tensor of shape (M, K) will produce a packed weight tensor of shape
    (M, K / 2).

    The packing implemented here is the same as the packing produced by
    backends/vulkan/_passes/int4_weight_only_quantizer.py
    """

    # Assert we got a properly quantized tensor.
    min, max = inp.min().item(), inp.max().item()
    assert (
        max <= 7 and min >= -8
    ), f"pack_4bit_weight_tensor: [min,max] out of [-8, 7] range, got [{min}, {max}]"

    # Assuming we have a 2d tensor
    if inp.ndim != 2:
        inp = inp.squeeze()
    assert (
        inp.ndim == 2
    ), f"pack_4bit_weight_tensor: expecting input tensor to be 2d, got {inp.ndim}"

    # pad ic
    if inp.shape[-1] % 2 != 0:
        inp = F.pad(input=inp, pad=(0, 1, 0, 0), mode="constant", value=0)

    # Shape after padding
    oc, ic = inp.shape
    assert ic % 2 == 0, "convert_to_qc4w: expecting ic to be even"

    # Adjust inp tensor for zp
    inp = inp.to(dtype=torch.uint8) + 8
    # Pack each 4-bit value into a single 8-bit value
    return inp[::, ::2] << 4 | inp[::, 1::2]


def make_combined_scales_and_zeros_tensor(
    scales: torch.Tensor, zeros: torch.Tensor
) -> torch.Tensor:
    """
    Given a scales and zeros tensor, create a combined tensor by stacking them into a
    single tensor.

    The scales and zeros tensors are expected to be 2D tensors of shape
    (OUTPUT_CHANNELS, NUM_GROUPS). The combined tensor will have the shape
    (NUM_GROUPS, OUTPUT_CHANNELS, 2).

    This is the scales and zeros format produced by
    backends/vulkan/_passes/int4_weight_only_quantizer.py, which in turn is the scales
    and zeros format expected by the _weight_int4pack_mm op in ATen.
    """
    scales_reshaped = scales.transpose(0, 1).unsqueeze(2)
    zeros_reshaped = zeros.transpose(0, 1).unsqueeze(2)

    zeros_scaled = zeros_reshaped * scales_reshaped * -1
    return torch.cat((scales_reshaped, zeros_scaled), dim=2)


def identify_wo_quantized_linear_io_nodes(  # noqa: C901
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: InternalMatch,
) -> Optional[List[torch.fx.Node]]:
    dequant_node = None
    # First, find the dequant node
    for node in match.nodes_map.values():
        if utils.is_dequant_node(node):
            dequant_node = node
            break

    if dequant_node is None:
        return None

    quantized_weight = dequant_node.args[0]
    quant_scales = dequant_node.args[2]
    quant_zeros = dequant_node.args[3]

    if not isinstance(quantized_weight, torch.fx.Node) or not is_param_node(
        ep, quantized_weight
    ):
        return None
    if not isinstance(quant_scales, torch.fx.Node) or not is_param_node(
        ep, quant_scales
    ):
        return None
    if not isinstance(quant_zeros, torch.fx.Node) or not is_param_node(ep, quant_zeros):
        return None

    input_nodes = match.placeholder_nodes
    if len(input_nodes) != 4:
        return None

    in_tensor_node = None
    for node in input_nodes:
        if node not in dequant_node.args:
            in_tensor_node = node
            break

    if in_tensor_node is None:
        return None

    output_nodes = match.returning_nodes

    if len(output_nodes) != 1:
        return None

    out_tensor_node = output_nodes[0]
    if not isinstance(out_tensor_node, torch.fx.Node):
        return None

    return [
        in_tensor_node,
        quantized_weight,
        quant_scales,
        quant_zeros,
        out_tensor_node,
    ]


# wo = "weight only"
def create_wo_quantized_linear_custom_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: InternalMatch,
):
    io_nodes = identify_wo_quantized_linear_io_nodes(ep, graph_module, match)
    if io_nodes is None:
        return

    assert len(io_nodes) == 5
    in_tensor, quantized_weight, quant_scales, quant_zeros, out_tensor = io_nodes

    quantized_weight_tensor = get_param_tensor(ep, quantized_weight)
    if not isinstance(quantized_weight_tensor, torch.Tensor):
        return
    packed_quantized_weight_tensor = pack_4bit_weight_tensor(quantized_weight_tensor)
    utils.update_program_state_dict(
        ep, quantized_weight.name, packed_quantized_weight_tensor
    )
    quantized_weight.meta["val"] = quantized_weight.meta["val"][:, ::2].to(torch.uint8)

    quant_scales_tensor = get_param_tensor(ep, quant_scales)
    quant_zeros_tensor = get_param_tensor(ep, quant_zeros)

    assert quantized_weight_tensor is not None
    assert quant_scales_tensor is not None
    assert quant_zeros_tensor is not None

    group_size = quantized_weight_tensor.shape[1] // quant_scales_tensor.shape[1]

    combined_scales_zeros_tensor = make_combined_scales_and_zeros_tensor(
        quant_scales_tensor, quant_zeros_tensor
    )

    combined_scales_zeros_name = f"{quantized_weight.name}_scales_zeros"
    graph_module.register_parameter(
        combined_scales_zeros_name, torch.nn.Parameter(combined_scales_zeros_tensor)
    )

    with graph_module.graph.inserting_before(out_tensor):
        combined_scales_zeros = graph_module.graph.get_attr(combined_scales_zeros_name)
        wo_qlinear = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.linear_weight_int4.default,
            args=(in_tensor, quantized_weight, group_size, combined_scales_zeros, 1),
        )

    if hasattr(out_tensor, "meta") and "val" in out_tensor.meta:
        wo_qlinear.meta["val"] = out_tensor.meta["val"]

    out_tensor.replace_all_uses_with(wo_qlinear)


class FusePatternsPass(ExportPass):
    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.program = exported_program

    def call(self, graph_module: torch.fx.GraphModule):
        total_replaced = 0

        total_replaced += fuse_pattern(
            self.program,
            graph_module,
            vk_patterns.get_rope_graphs(),
            create_rotary_emb_custom_op,
        )

        total_replaced += fuse_pattern(
            self.program,
            graph_module,
            vk_patterns.get_torchao_wo_quantized_linear_graphs(),
            create_wo_quantized_linear_custom_op,
        )

        if total_replaced > 0:
            graph_module.recompile()
            # Re-trace the graph
            graph_module = super().call(graph_module).graph_module

        return PassResult(graph_module, total_replaced > 0)
