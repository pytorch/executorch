# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, Tuple

import executorch.backends.vulkan.utils as utils
import torch

import torch.nn.functional as F

from executorch.backends.transforms.utils import get_param_tensor, is_param_node
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

#################
## linear_qcnw ##
#################


def matches_linear_qcnw_pattern(  # noqa: C901
    program: ExportedProgram, node: torch.fx.Node
) -> Optional[Tuple[torch.qscheme, int]]:
    """
    Checks if the nodes surrounding a linear node matches the pattern for weight only
    quantized linear, where the weight is quantized channelswise to n bits.

    If the graph pattern matches, then return a tuple of (quantization_method, nbits)
    describing the type of quantization used for the weights. Otherwise, return None.
    """
    if not utils.is_linear_node(node):
        return None

    input_node = node.args[0]
    weight_node = node.args[1]

    # Type checking
    if not isinstance(weight_node, torch.fx.Node):
        return None
    if not isinstance(input_node, torch.fx.Node):
        return None

    # The input arg should not be a dequant node; if it is, then it is indicative that
    # dynamically quantized linear should be used instead
    if utils.is_dequant_node(input_node):
        return None

    # The weight arg should be a dequant node dequantizing the quantized weight
    # Furthermore, the op expects per channel quantization of the weight
    if not utils.is_dequant_per_channel_node(weight_node):
        return None

    orig_weight = weight_node.args[0]
    zeros = weight_node.args[2]

    # Type checking
    if not isinstance(orig_weight, torch.fx.Node):
        return None
    if not is_param_node(program, orig_weight):
        return None
    if not isinstance(zeros, torch.fx.Node):
        return None
    if not is_param_node(program, zeros):
        return None

    zeros_tensor = get_param_tensor(program, zeros)
    if not isinstance(zeros_tensor, torch.Tensor):
        return None

    quant_method = torch.per_channel_affine
    # Check for symmetric quantization, where the zeros used for dequantization will
    # actually be all zeros.
    if torch.all(zeros_tensor == 0):
        quant_method = torch.per_channel_symmetric

    orig_weight_tensor = get_param_tensor(program, orig_weight)
    if not isinstance(orig_weight_tensor, torch.Tensor):
        return None
    # Sanity check the dtype of the quantized weight
    if orig_weight_tensor.dtype != torch.int8:
        return None

    quant_min = orig_weight_tensor.min().item()
    quant_max = orig_weight_tensor.max().item()
    # Determine the number of bits the weight has been quantized to
    if quant_min >= -8 and quant_max <= 7:
        return quant_method, 4
    elif quant_min >= -128 and quant_max <= 127:
        return quant_method, 8

    return None


def pack_4bit_weight_tensor(inp: torch.Tensor) -> torch.Tensor:
    """
    Given a 8-bit weight tensor containing values quantized to 4 bits, create a packed
    weight tensor by packing 2 4-bit values in one unsigned 8-bit value.

    An input weight tensor of shape (M, K) will produce a packed weight tensor of shape
    (M, K / 2).
    """

    # Assert we got a properly quantized tensor.
    min, max = inp.min().item(), inp.max().item()
    assert (
        max <= 7 and min >= -8
    ), f"convert_to_qc4w: [min,max] out of [-8, 7] range, got [{min}, {max}]"

    # Assuming we have a 2d tensor
    if inp.ndim != 2:
        inp = inp.squeeze()
    assert (
        inp.ndim == 2
    ), f"convert_to_qc4w: expecting input tensor to be 2d, got {inp.ndim}"

    # pad ic
    if inp.shape[-1] % 2 != 0:
        inp = F.pad(input=inp, pad=(0, 1, 0, 0), mode="constant", value=0)

    # Shape after padding
    oc, ic = inp.shape
    assert ic % 2 == 0, "convert_to_qc4w: expecting ic to be even"

    # Adjust inp tensor for zp
    inp = inp.to(dtype=torch.uint8) + 8

    # Prepare the Result tensor
    inp = inp.contiguous().view(-1)
    return (inp[::2] << 4 | inp[1::2]).view(oc, int(ic / 2))


def fuse_into_linear_qcnw_node(
    program: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    linear_node: torch.fx.Node,
    quant_method: torch.qscheme,
    nbits: int,
) -> None:
    """
    The weight_int8pack_mm operator represents a weight only quantized linear operator,
    where the weight tensor has been quantized channelswise to nbits bits.

      After the PT2E quantization flow, the expected graph pattern is

          dq_weight = dequantize(weight, scales)
          out = linear(activation, dq_weight, bias?)

      The goal of this function is to condense that sequence into

          out = quantized_linear(activation, dq_weight, scales)
          out = out + bias
    """
    activation = linear_node.args[0]
    dq_weight_node = linear_node.args[1]
    assert isinstance(activation, torch.fx.Node)
    assert isinstance(dq_weight_node, torch.fx.Node)

    bias = None
    if len(linear_node.args) > 2:
        bias = linear_node.args[2]
        assert isinstance(bias, torch.fx.Node)

    orig_weight = dq_weight_node.args[0]
    scale = dq_weight_node.args[1]

    # For 4 bit quantization, pack the weight tensor
    if nbits == 4:
        assert isinstance(orig_weight, torch.fx.Node)
        orig_weight_tensor = get_param_tensor(program, orig_weight)
        assert isinstance(orig_weight_tensor, torch.Tensor)
        packed_weight_tensor = pack_4bit_weight_tensor(orig_weight_tensor)
        utils.update_program_state_dict(
            program,
            orig_weight.name,
            packed_weight_tensor,
        )
        orig_weight.meta["val"] = orig_weight.meta["val"][:, ::2].to(torch.uint8)

    if nbits == 8 and quant_method == torch.per_channel_symmetric:
        op_target = exir_ops.edge.aten._weight_int8pack_mm.default
    elif nbits == 4 and quant_method == torch.per_channel_symmetric:
        op_target = exir_ops.edge.et_vk.linear_qcs4w.default
    else:
        raise NotImplementedError(
            "only 4 and 8 bits per channel symmetric quant supported for linear_qcnw"
        )

    with graph_module.graph.inserting_before(linear_node):
        weight_int8pack_mm_node = graph_module.graph.create_node(
            "call_function",
            op_target,
            (activation, orig_weight, scale),
        )
        if bias:
            add_node = graph_module.graph.create_node(
                "call_function",
                exir_ops.edge.aten.add.Tensor,
                (weight_int8pack_mm_node, bias),
            )
            linear_node.replace_all_uses_with(add_node)
        else:
            linear_node.replace_all_uses_with(weight_int8pack_mm_node)
        graph_module.graph.erase_node(linear_node)
        graph_module.graph.erase_node(dq_weight_node)


#########################
## linear_qta8a_qga4w ##
#########################


def _is_dequantize_affine_node(node: torch.fx.Node) -> bool:
    """Check if a node is a dequantize_affine operation."""
    return (
        node.op == "call_function"
        and node.target is not None
        and hasattr(node.target, "__name__")
        and "dequantize_affine" in getattr(node.target, "__name__", "")
    )


def _is_view_copy_node(node: torch.fx.Node) -> bool:
    """Check if a node is a view_copy operation."""
    return (
        node.op == "call_function"
        and node.target is not None
        and hasattr(node.target, "__name__")
        and "view_copy" in getattr(node.target, "__name__", "")
    )


def _validate_qta8a_qga4w_nodes(
    input_node: torch.fx.node.Argument, weight_node: torch.fx.node.Argument
) -> Optional[torch.fx.Node]:
    """
    Validate input and weight nodes for QTA8A_QGA4W pattern.
    Returns the actual input node (after handling view operations) or None if invalid.
    """
    # Type checking - ensure we have torch.fx.Node objects
    if not isinstance(weight_node, torch.fx.Node) or not isinstance(
        input_node, torch.fx.Node
    ):
        return None

    # Input may be preprocessed with a view node
    actual_input_node = input_node
    if _is_view_copy_node(input_node):
        actual_input_node = input_node.args[0]
        if not isinstance(actual_input_node, torch.fx.Node):
            return None

    # Check if input is dequantized with dequantize_affine (from dynamic quantization)
    if not _is_dequantize_affine_node(actual_input_node):
        return None

    # Check if weight is dequantized with dequantize_affine
    if not _is_dequantize_affine_node(weight_node):
        return None

    return actual_input_node


def _extract_weight_params(
    program: ExportedProgram, weight_node: torch.fx.Node
) -> Optional[Tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node]]:
    """Extract and validate weight parameters from dequantize_affine node."""
    # Get the original quantized weight and quantization parameters
    if len(weight_node.args) < 4:
        return None

    orig_weight = weight_node.args[0]
    weight_scales = weight_node.args[2]
    weight_zeros = weight_node.args[3]

    # Type checking
    if not isinstance(orig_weight, torch.fx.Node) or not is_param_node(
        program, orig_weight
    ):
        return None
    if not isinstance(weight_scales, torch.fx.Node) or not is_param_node(
        program, weight_scales
    ):
        return None
    if not isinstance(weight_zeros, torch.fx.Node) or not is_param_node(
        program, weight_zeros
    ):
        return None

    return orig_weight, weight_scales, weight_zeros


def _validate_4bit_quantization(weight_tensor: torch.Tensor) -> bool:
    """Check if weight tensor is quantized to 4 bits (values in [-8, 7] range)."""
    quant_min = weight_tensor.min().item()
    quant_max = weight_tensor.max().item()
    return quant_min >= -8 and quant_max <= 7


def _calculate_group_size(
    orig_weight_tensor: torch.Tensor, weight_scales_tensor: torch.Tensor
) -> Optional[int]:
    """Calculate and validate group size from weight and scales tensors."""
    out_features, in_features = orig_weight_tensor.shape

    if len(weight_scales_tensor.shape) != 2:
        return None

    scales_out_features, num_groups = weight_scales_tensor.shape

    if scales_out_features != out_features:
        return None

    group_size = in_features // num_groups
    if in_features % group_size != 0:
        return None

    return group_size


def matches_linear_qta8a_qga4w_pattern(
    program: ExportedProgram, node: torch.fx.Node
) -> Optional[Tuple[int, int]]:
    """
    Checks if the nodes surrounding a linear node matches the pattern for dynamic
    activation + grouped weight quantized linear (QTA8A_QGA4W).

    This pattern involves:
    1. Dynamic quantization of input activations (8-bit)
    2. Grouped quantization of weights (4-bit with group size)

    The expected pattern from Int8DynActInt4WeightQuantizer is:
        scale, zero_point = choose_qparams_affine(input)
        quantized_input = quantize_affine(input, scale, zero_point)
        dequantized_input = dequantize_affine(quantized_input, ...)
        dequantized_weight = dequantize_affine(weight, weight_scales, weight_zeros)
        output = linear(dequantized_input, dequantized_weight)

    If the pattern matches, return (group_size, weight_bits), otherwise None.
    """
    if not utils.is_linear_node(node):
        return None

    input_node = node.args[0]
    weight_node = node.args[1]

    # Validate nodes and get actual input node
    actual_input_node = _validate_qta8a_qga4w_nodes(input_node, weight_node)
    if actual_input_node is None:
        return None

    # Extract weight parameters
    if not isinstance(weight_node, torch.fx.Node):
        return None
    weight_params = _extract_weight_params(program, weight_node)
    if weight_params is None:
        return None

    orig_weight, weight_scales, weight_zeros = weight_params

    # Get tensors to analyze the quantization scheme
    orig_weight_tensor = get_param_tensor(program, orig_weight)
    weight_scales_tensor = get_param_tensor(program, weight_scales)
    weight_zeros_tensor = get_param_tensor(program, weight_zeros)

    if not isinstance(orig_weight_tensor, torch.Tensor):
        return None
    if not isinstance(weight_scales_tensor, torch.Tensor):
        return None
    if not isinstance(weight_zeros_tensor, torch.Tensor):
        return None

    # Check if weight is quantized to 4 bits
    if not _validate_4bit_quantization(orig_weight_tensor):
        return None

    # Calculate group size
    group_size = _calculate_group_size(orig_weight_tensor, weight_scales_tensor)
    if group_size is None:
        return None

    # Verify this is 4-bit grouped quantization
    weight_bits = 4

    return group_size, weight_bits


def fuse_into_linear_qta8a_qga4w_node(
    program: ExportedProgram,
    graph_module: torch.fx.GraphModule,
    linear_node: torch.fx.Node,
    group_size: int,
    weight_bits: int,
) -> None:
    """
    Fuse the dynamic activation + grouped weight quantized linear pattern into
    a single linear_qta8a_qga4w operator.

    The pattern:
        dequantized_input = dequantize_affine(quantized_input, block_size, scale, zero_point, ...)
        dequantized_weight = dequantize_affine(weight, block_size, weight_scales, weight_zeros, ...)
        output = linear(dequantized_input, dequantized_weight)

    Becomes:
        output = linear_qta8a_qga4w(quantized_input, input_scale, input_zero_point,
                                   weight, group_size, weight_scales, weight_zeros)
    """
    dq_input_node = linear_node.args[0]
    dq_weight_node = linear_node.args[1]

    assert isinstance(dq_input_node, torch.fx.Node)

    input_view_node = None
    # Input may be preprocessed with a view node
    if (
        dq_input_node.op == "call_function"
        and dq_input_node.target is not None
        and hasattr(dq_input_node.target, "__name__")
        and "view_copy" in getattr(dq_input_node.target, "__name__", "")
    ):
        input_view_node = dq_input_node
        dq_input_node = dq_input_node.args[0]
        assert isinstance(dq_input_node, torch.fx.Node)

    assert isinstance(dq_input_node, torch.fx.Node)
    assert isinstance(dq_weight_node, torch.fx.Node)

    # Get the quantized input and quantization parameters from the input dequantize_affine node
    # Args: (input, block_size, scale, zero_point, input_dtype, quant_min, quant_max, output_dtype)
    quantized_input = dq_input_node.args[0]
    input_scale = dq_input_node.args[2]  # scale is the 3rd argument
    input_zero_point = dq_input_node.args[3] if len(dq_input_node.args) > 3 else None

    # Get the weight and its quantization parameters from dequantize_affine
    # Args: (weight, block_size, weight_scales, weight_zeros, input_dtype, quant_min, quant_max, output_dtype)
    orig_weight = dq_weight_node.args[0]
    weight_scales = dq_weight_node.args[2]
    weight_zeros = dq_weight_node.args[3]

    # Pack the 4-bit weight tensor for efficient storage
    assert isinstance(orig_weight, torch.fx.Node)
    orig_weight_tensor = get_param_tensor(program, orig_weight)
    assert isinstance(orig_weight_tensor, torch.Tensor)
    packed_weight_tensor = pack_4bit_weight_tensor(orig_weight_tensor)
    utils.update_program_state_dict(
        program,
        orig_weight.name,
        packed_weight_tensor,
    )
    # Update the metadata to reflect the new packed shape
    orig_weight.meta["val"] = orig_weight.meta["val"][:, ::2].to(torch.uint8)

    # Create the linear_qta8a_qga4w node
    with graph_module.graph.inserting_before(linear_node):
        linear_qta8a_qga4w_node = graph_module.graph.create_node(
            "call_function",
            exir_ops.edge.et_vk.linear_qta8a_qga4w.default,
            (
                quantized_input,  # quantized input (int8)
                input_scale,  # mat1_scale
                input_zero_point,  # mat1_zero_point
                orig_weight,  # mat2_data (packed 4-bit weights)
                group_size,  # group_size (int)
                weight_scales,  # weight_scales
                weight_zeros,  # weight_zeros
            ),
        )

        # Replace the linear node with the new fused node
        linear_node.replace_all_uses_with(linear_qta8a_qga4w_node)

        # Erase nodes in the correct order (users first, then dependencies)
        graph_module.graph.erase_node(linear_node)
        if input_view_node is not None:
            graph_module.graph.erase_node(input_view_node)
        graph_module.graph.erase_node(dq_weight_node)
        graph_module.graph.erase_node(dq_input_node)


class FuseQuantizedOpsTransform(ExportPass):
    def __init__(self, exported_program: ExportedProgram) -> None:
        super().__init__()
        self.program = exported_program

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            # Check for linear_qcnw pattern (weight-only quantization)
            qcnw_details = matches_linear_qcnw_pattern(self.program, node)
            if qcnw_details is not None:
                qcnw_method, qcnw_nbits = qcnw_details
                fuse_into_linear_qcnw_node(
                    self.program, graph_module, node, qcnw_method, qcnw_nbits
                )
                continue

            # Check for linear_qta8a_qga4w pattern (dynamic activation + grouped weight quantization)
            qta8a_qga4w_details = None
            if qta8a_qga4w_details is not None:
                group_size, weight_bits = qta8a_qga4w_details
                fuse_into_linear_qta8a_qga4w_node(
                    self.program, graph_module, node, group_size, weight_bits
                )
                continue

        graph_module.recompile()
        dead_code_elimination_pass(graph_module)

        # Re-trace the graph since new nodes were (potentially) inserted
        graph_module = super().call(graph_module).graph_module
        return PassResult(graph_module, True)
