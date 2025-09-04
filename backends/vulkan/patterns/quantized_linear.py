# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import executorch.backends.vulkan.utils as utils

import torch
import torch.nn.functional as F

from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    get_param_tensor,
)

from executorch.backends.vulkan.patterns.pattern_registry import (
    PatternMatch,
    register_pattern_detector,
    register_pattern_replacement,
)

from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops

from torch.export.graph_signature import InputKind


class QuantizedLinearMatch(PatternMatch):
    def __init__(self, mm_node: torch.fx.Node) -> None:
        self.anchor_node = mm_node
        self.match_found = False
        self.all_nodes = [self.anchor_node]

        const_node, arg_chain = utils.trace_args_until_placeholder(
            self.anchor_node.args[1]
        )

        # mat2 is not a constant tensor - no match
        if const_node is None:
            return

        dequantize_weight_node = None
        # Search for a dequantize node in the arg chain of weight
        for node in arg_chain:
            if isinstance(node, torch.fx.Node) and utils.is_dequant_node(node):
                dequantize_weight_node = node
        # weight is not quantized - no match
        if dequantize_weight_node is None:
            return

        self.weight_node = const_node
        self.dequantize_weight_node = dequantize_weight_node
        self.all_nodes.extend(arg_chain)

        # By default, assume dequant node is from quantized_decomposed namespace
        scales_arg_idx = 1
        zeros_arg_idx = 2
        # torchao dequantize has a different function schema than quantized_decomposed
        if (
            self.dequantize_weight_node.target
            == exir_ops.edge.torchao.dequantize_affine.default
        ):
            scales_arg_idx = 2
            zeros_arg_idx = 3

        # Identify weight quantization parameter nodes
        self.weight_scales_node, arg_chain = utils.trace_args_until_placeholder(
            self.dequantize_weight_node.args[scales_arg_idx]
        )
        assert self.weight_scales_node is not None
        self.all_nodes.extend(arg_chain)

        self.weight_zeros_node, arg_chain = utils.trace_args_until_placeholder(
            self.dequantize_weight_node.args[zeros_arg_idx]
        )
        assert self.weight_zeros_node is not None
        self.all_nodes.extend(arg_chain)

        # Identify output node
        self.output_node = self.anchor_node

        # The implementation has a limitation that output channels must be a
        # multiple of 4. This is to ensure that data loads are aligned well with
        # texel boundaries. If this is not true, then don't match the pattern.
        out_channels = self.output_node.meta["val"].shape[-1]
        if out_channels % 4 != 0:
            return

        # Identify input node
        self.fp_input_node, self.quantize_input_node, dq_node = (
            utils.maybe_skip_q_dq_arg_chain(self.anchor_node.args[0])
        )
        assert self.fp_input_node is not None
        self.all_nodes.append(self.fp_input_node)

        # The implementation has a limitation that input channels must be a
        # multiple of 4. This is to ensure that data loads are aligned well with
        # texel boundaries. If this is not true, then don't match the pattern.
        in_channels = self.fp_input_node.meta["val"].shape[-1]
        if in_channels % 4 != 0:
            return

        # Identify bias node, if applicable
        self.bias_node = None
        if self.anchor_node.target == exir_ops.edge.aten.addmm.default:
            self.bias_node, arg_chain = utils.trace_args_until_placeholder(
                self.anchor_node.args[2]
            )
            assert self.bias_node is not None
            self.all_nodes.extend(arg_chain)

        # If input is not quantized, then we are done
        if self.quantize_input_node is None:
            self.match_found = True
            return

        self.input_scales_node = self.quantize_input_node.args[1]
        self.input_zeros_node = self.quantize_input_node.args[2]

        assert dq_node is not None
        self.all_nodes.extend(
            [
                self.quantize_input_node,
                dq_node,
            ]
        )

        self.match_found = True


linear_anchor_nodes = {
    exir_ops.edge.aten.linear.default,
    exir_ops.edge.aten.mm.default,
    exir_ops.edge.aten.addmm.default,
}


@register_pattern_detector("quantized_linear")
def find_quantized_linear_patterns(
    node: torch.fx.Node,
) -> Optional[QuantizedLinearMatch]:
    if node.target not in linear_anchor_nodes:
        return None

    matched_pattern = QuantizedLinearMatch(node)
    if matched_pattern.match_found:
        return matched_pattern

    return None


##
## Constant tensor manipulation
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


##
## Pattern Replacement
##


def make_linear_q4ga_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: QuantizedLinearMatch,
):
    weight_tensor = get_param_tensor(ep, match.weight_node)
    assert weight_tensor is not None

    assert match.weight_scales_node is not None
    weight_scales_tensor = get_param_tensor(ep, match.weight_scales_node)
    assert weight_scales_tensor is not None

    assert match.weight_zeros_node is not None
    weight_zeros_tensor = get_param_tensor(ep, match.weight_zeros_node)
    assert weight_zeros_tensor is not None

    packed_quantized_weight_tensor = pack_4bit_weight_tensor(weight_tensor)
    utils.update_program_state_dict(
        ep, match.weight_node.name, packed_quantized_weight_tensor
    )
    # Need to make sure corresponding FakeTensor has same size
    match.weight_node.meta["val"] = match.weight_node.meta["val"][:, ::2].to(
        torch.uint8
    )

    group_size = weight_tensor.shape[1] // weight_scales_tensor.shape[1]

    combined_scales_zeros_tensor = make_combined_scales_and_zeros_tensor(
        weight_scales_tensor, weight_zeros_tensor
    )

    combined_scales_zeros_name = f"{match.weight_node.name}_scales_zeros"
    graph_module.register_parameter(
        combined_scales_zeros_name, torch.nn.Parameter(combined_scales_zeros_tensor)
    )

    with graph_module.graph.inserting_before(match.output_node):
        combined_scales_zeros = graph_module.graph.get_attr(combined_scales_zeros_name)
        linear_q4ga_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.linear_weight_int4.default,
            args=(
                match.fp_input_node,
                match.weight_node,
                group_size,
                combined_scales_zeros,
                1,
            ),
        )

    linear_q4ga_node.meta["val"] = match.output_node.meta["val"]
    match.output_node.replace_all_uses_with(linear_q4ga_node)


def make_linear_q8ta_q8csw_custom_op(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: QuantizedLinearMatch,
):
    weight_tensor = get_param_tensor(ep, match.weight_node)
    assert weight_tensor is not None

    assert match.weight_scales_node is not None
    weight_scales_tensor = get_param_tensor(ep, match.weight_scales_node)
    assert weight_scales_tensor is not None

    assert match.weight_zeros_node is not None
    weight_zeros_tensor = get_param_tensor(ep, match.weight_zeros_node)
    assert weight_zeros_tensor is not None

    bias_tensor = None
    if match.bias_node is not None:
        bias_tensor = get_param_tensor(ep, match.bias_node)
        assert bias_tensor is not None

    first_graph_node = list(graph_module.graph.nodes)[0]
    with graph_module.graph.inserting_before(first_graph_node):
        weight_tensor_name = utils.get_tensor_name(ep, match.weight_node)
        # Pre-compute the weight sums which are needed to apply activation zero point
        # when using integer accumulation.
        sum_per_output_channel = weight_tensor.sum(dim=1).to(torch.float).contiguous()
        sums_name = weight_tensor_name + "_sums"
        # Sanitize the name
        sums_name = sums_name.replace(".", "_")

        weight_sums_node = create_constant_placeholder(
            exp_program=ep,
            graph=graph_module.graph,
            kind=InputKind.CONSTANT_TENSOR,
            name=sums_name,
            data=sum_per_output_channel,
        )

    with graph_module.graph.inserting_before(match.output_node):
        qlinear_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.linear_q8ta_q8csw.default,
            args=(
                match.fp_input_node,
                match.input_scales_node,
                match.input_zeros_node,
                match.weight_node,
                weight_sums_node,
                match.weight_scales_node,
            ),
        )

    qlinear_node.meta["val"] = match.output_node.meta["val"]
    match.output_node.replace_all_uses_with(qlinear_node)


@register_pattern_replacement("quantized_linear")
def replace_quantized_linear_patterns(
    ep: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    match: QuantizedLinearMatch,
):
    if match.quantize_input_node is None:
        make_linear_q4ga_op(ep, graph_module, match)
    else:
        make_linear_q8ta_q8csw_custom_op(ep, graph_module, match)
