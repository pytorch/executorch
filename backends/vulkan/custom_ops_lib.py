# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import executorch.backends.vulkan.patterns as vk_patterns
import torch.library
from torch._subclasses.fake_tensor import FakeTensor

namespace = "et_vk"
lib = torch.library.Library(namespace, "DEF")

#############
## prepack ##
#############


def prepack_impl(x: torch.Tensor):
    return x


name = "prepack"
lib.define(f"{name}(Tensor x) -> Tensor")
lib.impl(name, prepack_impl, "CompositeExplicitAutograd")
prepack_op = getattr(getattr(torch.ops, namespace), name)

#####################
## conv_with_clamp ##
#####################


def conv_with_clamp_impl(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    transposed=False,
    output_padding=0,
    groups=1,
    output_min=-float("inf"),
    output_max=float("inf"),
):
    return torch.clamp(
        torch.convolution(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ),
        output_min,
        output_max,
    )


name = "conv_with_clamp"
lib.define(
    f"{name}(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, Scalar? output_min, Scalar? output_max) -> Tensor"
)
lib.impl(name, conv_with_clamp_impl, "CompositeExplicitAutograd")
conv_with_clamp_op = getattr(getattr(torch.ops, namespace), name)

#########################
## conv_with_clamp.out ##
#########################


def conv_with_clamp_out_impl(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    transposed=False,
    output_padding=0,
    groups=1,
    output_min=-float("inf"),
    output_max=float("inf"),
    out=None,
):
    out = conv_with_clamp_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_min,
        output_max,
    )
    return out


name = "conv_with_clamp.out"
lib.define(
    f"{name}(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, Scalar? output_min, Scalar? output_max, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.impl(name, conv_with_clamp_out_impl, "CompositeExplicitAutograd")

#################
## grid_priors ##
#################


# The dimension of x should be larger than 1
def grid_priors_impl(
    x,
    stride,
    offset,
):
    height, width = x.shape[-2:]
    # Need to specify device of torch.arange to avoid executorch exporting error
    shift_x = (torch.arange(0, width, device=x.device) + offset) * stride
    shift_y = (torch.arange(0, height, device=x.device) + offset) * stride
    # Need to specify indexing parameter ('ij' is the default value) to avoid executorch exporting error
    shift_xx, shift_yy = torch.meshgrid([shift_y, shift_x], indexing="ij")
    shift_xx = shift_xx.reshape(-1)
    shift_yy = shift_yy.reshape(-1)
    shifts = torch.stack((shift_yy, shift_xx), dim=-1)
    return shifts


name = "grid_priors"
lib.define(f"{name}(Tensor self, int stride, float offset) -> Tensor")
lib.impl(name, grid_priors_impl, "CompositeExplicitAutograd")
grid_priors_op = getattr(getattr(torch.ops, namespace), name)


# When lowering to executorch, ops are converted from default to out variant. Hence, custom ops define both variants.
def grid_priors_out_impl(
    x,
    stride,
    offset,
    out,
):
    out = grid_priors_impl(x, stride, offset)
    return out


name = "grid_priors_out"
lib.define(
    f"{name}(Tensor self, int stride, float offset, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.impl(name, grid_priors_out_impl, "CompositeExplicitAutograd")

##################
## linear_qcs4w ##
##################


def linear_qcs4w(
    x: torch.Tensor,
    weights_4x2: torch.Tensor,
    scales: torch.Tensor,
):
    original_x_shape = x.shape
    x = x.reshape(-1, original_x_shape[-1])

    unpacked_weights_shape = weights_4x2.shape
    out_features = unpacked_weights_shape[0]
    in_features = unpacked_weights_shape[1]

    weights_unpacked = torch.empty(
        (out_features, in_features * 2), dtype=torch.int8, device=weights_4x2.device
    )

    weights_unpacked[:, ::2] = weights_4x2 >> 4
    weights_unpacked[:, 1::2] = weights_4x2 & 0x0F

    n_bit = 8
    quant_min = -(2 ** (n_bit - 1))
    quant_max = 2 ** (n_bit - 1) - 1
    dq_weights = torch.ops.quantized_decomposed.dequantize_per_channel(
        weights_unpacked,
        scales,
        None,
        0,
        quant_min,
        quant_max,
        torch.int8,
    )

    out = torch.nn.functional.linear(x, dq_weights)
    out_shape = original_x_shape[:-1] + (out_features,)
    return out.reshape(out_shape)


name = "linear_qcs4w"
lib.define(f"{name}(Tensor self, Tensor weight, Tensor scales) -> Tensor")
lib.impl(name, linear_qcs4w, "CompositeExplicitAutograd")
linear_qc4w_op = getattr(getattr(torch.ops, namespace), name)

##################
## linear_q4gsw ##
##################


def unpack_4bit_weight_tensor(
    packed_weight_tensor: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """
    Reverses the packing performed in quantized_linear.pack_4bit_weight_tensor
    """
    # Each packed byte contains two 4-bit values: high nibble and low nibble
    K, N_half = packed_weight_tensor.shape
    N = N_half * 2

    # Unpack high and low nibbles
    high_nibble = (packed_weight_tensor >> 4) & 0x0F
    low_nibble = packed_weight_tensor & 0x0F

    # Stack to shape (K, N)
    unpacked = torch.empty(
        (K, N), dtype=torch.uint8, device=packed_weight_tensor.device
    )
    unpacked[:, ::2] = low_nibble
    unpacked[:, 1::2] = high_nibble

    # Undo the +8 offset and convert to signed 4-bit range [-8, 7]
    unpacked = unpacked.to(torch.int8) - 8

    in_channels = x.shape[-1]
    # Undo any padding that may have been added to input channels
    if in_channels != unpacked.shape[-1]:
        return unpacked[:, :in_channels]

    return unpacked


def linear_q4gsw(
    x: torch.Tensor,
    weights: torch.Tensor,
    weight_scales: torch.Tensor,
    group_size: int,
    bias: Optional[torch.Tensor] = None,
):
    # Unpack the packed weights
    weights = unpack_4bit_weight_tensor(weights, x)

    # Un-transpose the weight scales
    weight_scales = weight_scales.transpose(0, 1)
    weight_zeros = torch.zeros_like(weight_scales, dtype=torch.int32)

    weights = torch.ops.torchao.dequantize_affine(
        weights, [1, group_size], weight_scales, weight_zeros, torch.int8, -8, 7
    )

    out = torch.nn.functional.linear(x, weights, bias)
    return out


def linear_dq8ca_q4gsw(
    x: torch.Tensor,
    input_scale: torch.Tensor,
    input_zero_point: torch.Tensor,
    weights: torch.Tensor,
    weight_sums: torch.Tensor,
    weight_scales: torch.Tensor,
    group_size: int,
    bias: Optional[torch.Tensor] = None,
):
    return linear_q4gsw(x, weights, weight_scales, group_size, bias)


name = "linear_q4gsw"
lib.define(
    f"""
            {name}(
                Tensor self,
                Tensor weights,
                Tensor weight_scales,
                int group_size,
                Tensor? bias = None) -> Tensor
            """
)
lib.impl(name, linear_q4gsw, "CompositeExplicitAutograd")
linear_qc4w_op = getattr(getattr(torch.ops, namespace), name)


# Backward of linear_q4gsw wrt input (for on-device LoRA training through a frozen
# 4-bit base): d_x = d_out @ dequant(W). Reference impl extracts dequant(W) via the
# forward on an identity so it is layout-agnostic; the runtime dispatches this op to
# the tiled q4gsw_backward WGSL kernel (contracts over N).
def linear_q4gsw_backward_impl(
    d_out: torch.Tensor,
    weights: torch.Tensor,
    weight_scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    in_features = int(weights.shape[1]) * 2
    eye = torch.eye(in_features, dtype=d_out.dtype, device=d_out.device)
    w_t = linear_q4gsw(eye, weights, weight_scales, group_size)  # [in, out]
    return d_out @ w_t.t()  # [M, out] @ [out, in] = [M, in]


def linear_q4gsw_backward_meta(
    d_out: torch.Tensor,
    weights: torch.Tensor,
    weight_scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    return d_out.new_empty(d_out.shape[:-1] + (int(weights.shape[1]) * 2,))


name = "linear_q4gsw_backward"
lib.define(
    f"{name}(Tensor d_out, Tensor weights, Tensor weight_scales, int group_size) -> Tensor"
)
lib.impl(name, linear_q4gsw_backward_impl, "CompositeExplicitAutograd")
lib.impl(name, linear_q4gsw_backward_meta, "Meta")
linear_q4gsw_backward_op = getattr(getattr(torch.ops, namespace), name)


def linear_q4gsw_setup_context(ctx, inputs, output) -> None:
    _x, weights, weight_scales, group_size, _bias = inputs
    ctx.save_for_backward(weights, weight_scales)
    ctx.group_size = group_size


def linear_q4gsw_backward(ctx, grad_out):
    weights, weight_scales = ctx.saved_tensors
    d_x = torch.ops.et_vk.linear_q4gsw_backward(
        grad_out, weights, weight_scales, ctx.group_size
    )
    return d_x, None, None, None, None  # grads for (x, weights, scales, group_size, bias)


torch.library.register_autograd(
    f"{namespace}::linear_q4gsw",
    linear_q4gsw_backward,
    setup_context=linear_q4gsw_setup_context,
)

name = "linear_dq8ca_q4gsw"
lib.define(
    f"""
            {name}(
                Tensor input,
                Tensor input_scales,
                Tensor input_zp,
                Tensor weights,
                Tensor weight_sums,
                Tensor weight_scales,
                int group_size,
                Tensor? bias = None) -> Tensor
            """
)
lib.impl(name, linear_dq8ca_q4gsw, "CompositeExplicitAutograd")
linear_dq8ca_q4gsw_op = getattr(getattr(torch.ops, namespace), name)

#################
## qaqw_linear ##
#################


def linear_q8ta_q8csw(
    x: torch.Tensor,
    input_scale: float,
    input_zero_point: int,
    weights: torch.Tensor,
    weight_sums: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
):
    weight_zeros = torch.zeros_like(weight_scales, dtype=torch.int32)
    weights = torch.ops.quantized_decomposed.dequantize_per_channel(
        weights,
        weight_scales,
        weight_zeros,
        0,
        -127,
        127,
        torch.int8,
    )

    # Perform linear operation
    out = torch.nn.functional.linear(x, weights)
    if bias is not None:
        out = out + bias

    return out


name = "linear_q8ta_q8csw"
lib.define(
    f"""
    {name}(
        Tensor x,
        float input_scale,
        int input_zero_point,
        Tensor weights,
        Tensor weight_sums,
        Tensor weight_scales,
        Tensor? bias = None) -> Tensor
    """
)
lib.impl(name, linear_q8ta_q8csw, "CompositeExplicitAutograd")
qa_q8csw_linear = getattr(getattr(torch.ops, namespace), name)

##################
## q8ta_linear ##
##################


def q8ta_linear(
    x: torch.Tensor,
    input_scale: float,
    input_zero_point: int,
    weights: torch.Tensor,
    weight_sums: torch.Tensor,
    weight_scales: torch.Tensor,
    output_scale: float,
    output_zero_point: int,
    bias: Optional[torch.Tensor] = None,
    activation: str = "none",
):
    weight_zeros = torch.zeros_like(weight_scales, dtype=torch.int32)
    weights = torch.ops.quantized_decomposed.dequantize_per_channel(
        weights,
        weight_scales,
        weight_zeros,
        0,
        -127,
        127,
        torch.int8,
    )

    x = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x, input_scale, input_zero_point, -128, 127, x.dtype
    )

    out = torch.nn.functional.linear(x, weights)
    if bias is not None:
        out = out + bias[: out.shape[-1]]

    if activation == "relu":
        out = torch.nn.functional.relu(out)

    out = torch.ops.quantized_decomposed.quantize_per_tensor(
        out, output_scale, output_zero_point, -128, 127, torch.int8
    )

    return out


name = "q8ta_linear"
lib.define(
    f"""
    {name}(
        Tensor x,
        float input_scale,
        int input_zero_point,
        Tensor weights,
        Tensor weight_sums,
        Tensor weight_scales,
        float output_scale,
        int output_zero_point,
        Tensor? bias = None,
        str activation = "none") -> Tensor
    """
)
lib.impl(name, q8ta_linear, "CompositeExplicitAutograd")
q8ta_linear_op = getattr(getattr(torch.ops, namespace), name)

#######################
## q8ta_linear_gemv ##
#######################


def q8ta_linear_gemv(
    x: torch.Tensor,
    input_scale: float,
    input_zero_point: int,
    weights: torch.Tensor,
    weight_sums: torch.Tensor,
    weight_scales: torch.Tensor,
    output_scale: float,
    output_zero_point: int,
    bias: Optional[torch.Tensor] = None,
    activation: str = "none",
):
    weight_zeros = torch.zeros_like(weight_scales, dtype=torch.int32)
    weights = torch.ops.quantized_decomposed.dequantize_per_channel(
        weights,
        weight_scales,
        weight_zeros,
        0,
        -127,
        127,
        torch.int8,
    )

    x = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x, input_scale, input_zero_point, -128, 127, x.dtype
    )

    out = torch.nn.functional.linear(x, weights)
    if bias is not None:
        out = out + bias[: out.shape[-1]]

    if activation == "relu":
        out = torch.nn.functional.relu(out)

    out = torch.ops.quantized_decomposed.quantize_per_tensor(
        out, output_scale, output_zero_point, -128, 127, torch.int8
    )

    return out


name = "q8ta_linear_gemv"
lib.define(
    f"""
    {name}(
        Tensor x,
        float input_scale,
        int input_zero_point,
        Tensor weights,
        Tensor weight_sums,
        Tensor weight_scales,
        float output_scale,
        int output_zero_point,
        Tensor? bias = None,
        str activation = "none") -> Tensor
    """
)
lib.impl(name, q8ta_linear_gemv, "CompositeExplicitAutograd")
q8ta_linear_gemv_op = getattr(getattr(torch.ops, namespace), name)

###################
## q8ta_conv2d_* ##
###################


def q8ta_conv2d(
    x: torch.Tensor,
    input_scale: float,
    input_zero_point: int,
    weights: torch.Tensor,
    weight_sums: torch.Tensor,
    weight_scales: torch.Tensor,
    output_scale: float,
    output_zero_point: int,
    bias: Optional[torch.Tensor],
    kernel_size: list,
    stride: list,
    padding: list,
    dilation: list,
    groups: int,
    activation: str,
):
    x = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x, input_scale, input_zero_point, -128, 127, x.dtype
    )

    # Calculate weight dimensions
    OC = weights.shape[0]
    assert OC % groups == 0, "Output channels must be divisible by groups"
    IC_per_group = int(x.shape[1] / groups)
    K_h, K_w = kernel_size[0], kernel_size[1]

    orig_weight_K_dim = K_h * K_w * IC_per_group
    # Remove any padding added to in_features dim to align to a multiple of 4
    if weights.shape[-1] > orig_weight_K_dim:
        weights = weights[:, :orig_weight_K_dim]

    # Remove any padding added to output channels dim to align to a multiple of 4
    if weight_scales.shape[0] > OC:
        weight_scales = weight_scales[:OC]
        if bias is not None:
            bias = bias[:OC]

    # Reshape to original 4D format (OC, IC, H, W)
    weights = weights.view(OC, IC_per_group, K_h, K_w)

    weight_zeros = torch.zeros_like(weight_scales, dtype=torch.int32)
    # Dequantize weights
    weights = torch.ops.quantized_decomposed.dequantize_per_channel(
        weights,
        weight_scales,
        weight_zeros,
        0,  # axis=0 for output channel quantization
        -127,
        127,
        torch.int8,
    )

    # Perform convolution
    out = torch.nn.functional.conv2d(
        x, weights, bias, stride, padding, dilation, groups
    )

    if activation == "relu":
        out = torch.nn.functional.relu(out)

    out = torch.ops.quantized_decomposed.quantize_per_tensor(
        out, output_scale, output_zero_point, -128, 127, torch.int8
    )

    return out


name = "q8ta_conv2d"
lib.define(
    f"""
    {name}(
        Tensor x,
        float input_scale,
        int input_zero_point,
        Tensor weights,
        Tensor weight_sums,
        Tensor weight_scales,
        float output_scale,
        int output_zero_point,
        Tensor? bias,
        SymInt[] kernel_size,
        SymInt[] stride,
        SymInt[] padding,
        SymInt[] dilation,
        SymInt groups,
        str activation) -> Tensor
    """
)
lib.impl(name, q8ta_conv2d, "CompositeExplicitAutograd")
q8ta_conv2d_op = getattr(getattr(torch.ops, namespace), name)


name = "q8ta_conv2d_pw"
lib.define(
    f"""
    {name}(
        Tensor x,
        float input_scale,
        int input_zero_point,
        Tensor weights,
        Tensor weight_sums,
        Tensor weight_scales,
        float output_scale,
        int output_zero_point,
        Tensor? bias,
        SymInt[] kernel_size,
        SymInt[] stride,
        SymInt[] padding,
        SymInt[] dilation,
        SymInt groups,
        str activation) -> Tensor
    """
)
lib.impl(name, q8ta_conv2d, "CompositeExplicitAutograd")
q8ta_conv2d_pw_op = getattr(getattr(torch.ops, namespace), name)


def q8ta_conv2d_dw(
    x: torch.Tensor,
    input_scale: float,
    input_zero_point: int,
    weights: torch.Tensor,
    weight_sums: torch.Tensor,
    weight_scales: torch.Tensor,
    output_scale: float,
    output_zero_point: int,
    bias: Optional[torch.Tensor],
    kernel_size: list,
    stride: list,
    padding: list,
    dilation: list,
    groups: int,
    activation: str,
):
    x = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x, input_scale, input_zero_point, -128, 127, x.dtype
    )

    # Restore weight to original data layout
    K_h, K_w, OC = weights.shape
    weights = weights.permute(2, 0, 1).reshape(OC, 1, K_h, K_w)

    weight_zeros = torch.zeros_like(weight_scales, dtype=torch.int32)
    # Dequantize weights
    weights = torch.ops.quantized_decomposed.dequantize_per_channel(
        weights,
        weight_scales,
        weight_zeros,
        0,  # axis=0 for output channel quantization
        -127,
        127,
        torch.int8,
    )

    # Perform convolution
    out = torch.nn.functional.conv2d(
        x, weights, bias, stride, padding, dilation, groups
    )

    if activation == "relu":
        out = torch.nn.functional.relu(out)

    out = torch.ops.quantized_decomposed.quantize_per_tensor(
        out, output_scale, output_zero_point, -128, 127, torch.int8
    )

    return out


name = "q8ta_conv2d_dw"
lib.define(
    f"""
    {name}(
        Tensor x,
        float input_scale,
        int input_zero_point,
        Tensor weights,
        Tensor weight_sums,
        Tensor weight_scales,
        float output_scale,
        int output_zero_point,
        Tensor? bias,
        SymInt[] kernel_size,
        SymInt[] stride,
        SymInt[] padding,
        SymInt[] dilation,
        SymInt groups,
        str activation) -> Tensor
    """
)
lib.impl(name, q8ta_conv2d_dw, "CompositeExplicitAutograd")
conv2d_q8ta_q8csw_dw_op = getattr(getattr(torch.ops, namespace), name)


def q8ta_conv2d_transposed(
    x: torch.Tensor,
    input_scale: float,
    input_zero_point: int,
    weights: torch.Tensor,
    weight_sums: torch.Tensor,
    weight_scales: torch.Tensor,
    output_scale: float,
    output_zero_point: int,
    bias: Optional[torch.Tensor],
    kernel_size: list,
    stride: list,
    padding: list,
    output_padding: list,
    dilation: list,
    groups: int,
    activation: str,
):
    x = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x, input_scale, input_zero_point, -128, 127, x.dtype
    )

    OC = weights.shape[0]
    IC_per_group = int(x.shape[1] / groups)
    K_h, K_w = kernel_size[0], kernel_size[1]

    orig_weight_K_dim = K_h * K_w * IC_per_group
    if weights.shape[-1] > orig_weight_K_dim:
        weights = weights[:, :orig_weight_K_dim]

    if weight_scales.shape[0] > OC:
        weight_scales = weight_scales[:OC]
        if bias is not None:
            bias = bias[:OC]

    # Reshape to (OC, IC_per_group, K_h, K_w) then transpose to
    # (IC_per_group * groups, OC_per_group, K_h, K_w) for conv_transpose2d
    weights = weights.view(OC, IC_per_group, K_h, K_w)
    OC_per_group = OC // groups
    weights = (
        weights.view(groups, OC_per_group, IC_per_group, K_h, K_w)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .view(IC_per_group * groups, OC_per_group, K_h, K_w)
    )

    weight_zeros = torch.zeros_like(weight_scales, dtype=torch.int32)
    # Dequantize per OC channel. For transposed weight (IC, OC_per_group, KH, KW),
    # OC is at axis=1.
    weights = torch.ops.quantized_decomposed.dequantize_per_channel(
        weights,
        weight_scales[:OC_per_group].repeat(groups) if groups > 1 else weight_scales,
        weight_zeros[:OC_per_group].repeat(groups) if groups > 1 else weight_zeros,
        1,
        -127,
        127,
        torch.int8,
    )

    out = torch.nn.functional.conv_transpose2d(
        x, weights, bias, stride, padding, output_padding, groups, dilation
    )

    if activation == "relu":
        out = torch.nn.functional.relu(out)

    out = torch.ops.quantized_decomposed.quantize_per_tensor(
        out, output_scale, output_zero_point, -128, 127, torch.int8
    )

    return out


name = "q8ta_conv2d_transposed"
lib.define(
    f"""
    {name}(
        Tensor x,
        float input_scale,
        int input_zero_point,
        Tensor weights,
        Tensor weight_sums,
        Tensor weight_scales,
        float output_scale,
        int output_zero_point,
        Tensor? bias,
        SymInt[] kernel_size,
        SymInt[] stride,
        SymInt[] padding,
        SymInt[] output_padding,
        SymInt[] dilation,
        SymInt groups,
        str activation) -> Tensor
    """
)
lib.impl(name, q8ta_conv2d_transposed, "CompositeExplicitAutograd")
q8ta_conv2d_transposed_op = getattr(getattr(torch.ops, namespace), name)

######################
## apply_rotary_emb ##
######################


def apply_rotary_emb_impl(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
):
    pattern = vk_patterns.RotaryEmbeddingPattern()
    return pattern.forward(xq, xk, freqs_cos, freqs_sin)


name = "apply_rotary_emb"
lib.define(
    f"{name}(Tensor xq, Tensor xk, Tensor freqs_cos, Tensor freqs_sin) -> (Tensor, Tensor)"
)
lib.impl(name, apply_rotary_emb_impl, "CompositeExplicitAutograd")
apply_rotary_emb_op = getattr(getattr(torch.ops, namespace), name)

#########################
## apply_rotary_emb_hf ##
#########################


def apply_rotary_emb_hf_impl(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    start_pos: int,
):
    seq_len = xq.shape[1]
    freqs_cos = freqs_cos[start_pos : start_pos + seq_len]
    freqs_sin = freqs_sin[start_pos : start_pos + seq_len]
    pattern = vk_patterns.HfRotaryEmbeddingPattern()
    return pattern.forward(xq, xk, freqs_cos, freqs_sin)


name = "apply_rotary_emb_hf"
lib.define(
    f"{name}(Tensor xq, Tensor xk, Tensor freqs_cos, Tensor freqs_sin, SymInt start_pos) -> (Tensor, Tensor)"
)
lib.impl(name, apply_rotary_emb_hf_impl, "CompositeExplicitAutograd")
apply_rotary_emb_hf_op = getattr(getattr(torch.ops, namespace), name)

##################################
## apply_rotary_emb_interleaved ##
##################################


def apply_rotary_emb_interleaved_impl(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    # EdgeTAM's pair-interleaved complex-number RoPE.
    #   x:         [B, N, C] with (real, imag) pairs interleaved along C
    #   freqs_cis: any rank whose flattened layout is [N, C]. Commonly 2D
    #              [N, C] or 4D [1, N, C/2, 2] from
    #              `torch.view_as_real(...).unsqueeze(0)`. The (cos, sin)
    #              pairs are interleaved along the innermost axis in the
    #              flattened view.
    # Semantically equivalent to:
    #   freqs_cis.reshape(N, C // 2, 2) -> (cos, sin)
    #   out[2k]   = x[2k] * cos[k] - x[2k+1] * sin[k]
    #   out[2k+1] = x[2k] * sin[k] + x[2k+1] * cos[k]
    B, N, C = x.shape
    a_real, a_imag = x.view(B, N, C // 2, 2).unbind(-1)
    # Use reshape so callers may pass freqs_cis at any rank.
    cs = freqs_cis.reshape(N, C // 2, 2)
    b_real, b_imag = cs[..., 0], cs[..., 1]
    out = torch.stack(
        (a_real * b_real - a_imag * b_imag, a_real * b_imag + a_imag * b_real),
        dim=-1,
    )
    return out.view(B, N, C)


def apply_rotary_emb_interleaved_meta(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    # Meta kernel: shape-only. Keeps the op opaque during torch.export (no
    # inlining of view/reshape calls into the exported graph) and does not
    # constrain the rank of freqs_cis — any shape with N * C elements is
    # accepted by the Vulkan dispatcher.
    return torch.empty_like(x)


name = "apply_rotary_emb_interleaved"
lib.define(f"{name}(Tensor x, Tensor freqs_cis) -> Tensor")
# CPU kernel preserves eager-mode reference semantics.
lib.impl(name, apply_rotary_emb_interleaved_impl, "CPU")
# Meta kernel keeps the op opaque in the exported graph.
lib.impl(name, apply_rotary_emb_interleaved_meta, "Meta")
apply_rotary_emb_interleaved_op = getattr(getattr(torch.ops, namespace), name)

########################
## q8ta_add ##
########################


def q8ta_add_impl(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    input_a_scale: float,
    input_a_zero_point: int,
    input_b_scale: float,
    input_b_zero_point: int,
    output_scale: float,
    output_zero_point: int,
    alpha: float,
):
    # Dequantize inputs to float
    dequant_a = torch.ops.quantized_decomposed.dequantize_per_tensor(
        input_a, input_a_scale, input_a_zero_point, -128, 127, input_a.dtype
    )
    dequant_b = torch.ops.quantized_decomposed.dequantize_per_tensor(
        input_b, input_b_scale, input_b_zero_point, -128, 127, input_b.dtype
    )

    # Perform addition with alpha scaling
    result = dequant_a + alpha * dequant_b

    # Quantize the result back to int8
    quantized_result = torch.ops.quantized_decomposed.quantize_per_tensor(
        result, output_scale, output_zero_point, -128, 127, torch.int8
    )

    return quantized_result


name = "q8ta_add"
lib.define(
    f"{name}(Tensor input_a, Tensor input_b, float input_a_scale, int input_a_zero_point, float input_b_scale, int input_b_zero_point, float output_scale, int output_zero_point, float alpha) -> Tensor"
)
lib.impl(name, q8ta_add_impl, "CompositeExplicitAutograd")
q8ta_add_op = getattr(getattr(torch.ops, namespace), name)

########################
## q8ta_relu ##
########################


def q8ta_relu_impl(
    input: torch.Tensor,
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
):
    # Dequantize input to float
    dequant = torch.ops.quantized_decomposed.dequantize_per_tensor(
        input, input_scale, input_zero_point, -128, 127, input.dtype
    )

    # Apply ReLU
    result = torch.nn.functional.relu(dequant)

    # Quantize the result back to int8
    quantized_result = torch.ops.quantized_decomposed.quantize_per_tensor(
        result, output_scale, output_zero_point, -128, 127, torch.int8
    )

    return quantized_result


name = "q8ta_relu"
lib.define(
    f"{name}(Tensor input, float input_scale, int input_zero_point, float output_scale, int output_zero_point) -> Tensor"
)
lib.impl(name, q8ta_relu_impl, "CompositeExplicitAutograd")
q8ta_relu_op = getattr(getattr(torch.ops, namespace), name)

###########################
## q8ta_pixel_shuffle    ##
###########################


def q8ta_pixel_shuffle_impl(
    input: torch.Tensor,
    input_scale: float,
    input_zero_point: int,
    output_inv_scale: float,
    output_zero_point: int,
    upscale_factor: int,
):
    # Reference Python impl for op registration. The runtime kernel does a
    # fused byte-shuffle (and optional requantize when scales differ).
    output_scale = 1.0 / output_inv_scale
    dequant = torch.ops.quantized_decomposed.dequantize_per_tensor(
        input, input_scale, input_zero_point, -128, 127, input.dtype
    )
    shuffled = torch.nn.functional.pixel_shuffle(dequant, upscale_factor)
    requantized = torch.ops.quantized_decomposed.quantize_per_tensor(
        shuffled, output_scale, output_zero_point, -128, 127, torch.int8
    )
    return requantized


name = "q8ta_pixel_shuffle"
lib.define(
    f"{name}(Tensor input, float input_scale, int input_zero_point, float output_inv_scale, int output_zero_point, int upscale_factor) -> Tensor"
)
lib.impl(name, q8ta_pixel_shuffle_impl, "CompositeExplicitAutograd")
q8ta_pixel_shuffle_op = getattr(getattr(torch.ops, namespace), name)

########################
## embedding_q4gsw ##
########################


def embedding_q4gsw_impl(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    group_size: int,
    indices: torch.Tensor,
    is_linear_weight: bool = False,
) -> torch.Tensor:
    # Unpack 4-bit values from packed uint8 tensor
    # Packing convention: packed_byte = (even_val + 8) << 4 | (odd_val + 8)
    high = (weight >> 4).to(torch.int8) - 8
    low = (weight & 0xF).to(torch.int8) - 8
    if is_linear_weight:
        unpacked = torch.stack([low, high], dim=-1).reshape(weight.shape[0], -1)
    else:
        unpacked = torch.stack([high, low], dim=-1).reshape(weight.shape[0], -1)
    # Dequantize using per-group scales
    num_groups = weight_scales.shape[1] if weight_scales.dim() > 1 else 1
    unpacked_groups = unpacked.reshape(weight.shape[0], num_groups, group_size)
    scales = (
        weight_scales.unsqueeze(-1)
        if weight_scales.dim() > 1
        else weight_scales.reshape(1, 1, 1)
    )
    dequantized = unpacked_groups.float() * scales.float()
    dequantized = dequantized.reshape(weight.shape[0], -1)
    return torch.nn.functional.embedding(indices, dequantized)


name = "embedding_q4gsw"
lib.define(
    f"{name}(Tensor weight, Tensor weight_scales, int group_size, Tensor indices, bool is_linear_weight = False) -> Tensor"
)
lib.impl(name, embedding_q4gsw_impl, "CompositeExplicitAutograd")
embedding_q4gsw_op = getattr(getattr(torch.ops, namespace), name)

#############################
## select_as_symint ##
#############################


def select_as_symint_impl(x: torch.Tensor, dim: int, index: int):
    assert isinstance(x, FakeTensor)
    return x.fake_mode.shape_env.create_unbacked_symint()


name = "select_as_symint"
lib.define(f"{name}(Tensor x, int dim, int index) -> SymInt")
lib.impl(name, select_as_symint_impl, "Meta")
select_as_symint_op = getattr(getattr(torch.ops, namespace), name)

##########
## sdpa ##
##########


def sdpa_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
):
    if scale is None:
        scale = 1.0 / (q.size(-1) ** 0.5)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


name = "sdpa"
lib.define(
    f"{name}(Tensor q, Tensor k, Tensor v, Tensor? attn_mask = None, float? scale = None) -> Tensor"
)
lib.impl(name, sdpa_impl, "CompositeExplicitAutograd")
sdpa_op = getattr(getattr(torch.ops, namespace), name)

################
## rms_norm ##
################


def rms_norm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    input_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(variance + eps)
    return (x_normed * weight.float()).to(input_dtype)


name = "rms_norm"
lib.define(f"{name}(Tensor x, Tensor weight, float eps) -> Tensor")
lib.impl(name, rms_norm_impl, "CompositeExplicitAutograd")
rms_norm_op = getattr(getattr(torch.ops, namespace), name)


# STE weight gradient d_out^T @ x through the frozen 4-bit linear_q4gsw base.
def linear_q4gsw_dw_impl(
    d_out: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    return d_out.reshape(-1, d_out.shape[-1]).t() @ x.reshape(-1, x.shape[-1])


def linear_q4gsw_dw_meta(
    d_out: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    return d_out.new_empty((d_out.shape[-1], x.shape[-1]))


name = "linear_q4gsw_dw"
lib.define(f"{name}(Tensor d_out, Tensor x) -> Tensor")
lib.impl(name, linear_q4gsw_dw_impl, "CompositeExplicitAutograd")
lib.impl(name, linear_q4gsw_dw_meta, "Meta")
linear_q4gsw_dw_op = getattr(getattr(torch.ops, namespace), name)



##################
## q4gsw_requant ##
##################


# STE re-quant of fp32 latent weights into the frozen-scale 4-bit codes.
def q4gsw_requant_impl(
    latent: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    n, k = latent.shape
    group_idx = torch.arange(k, device=latent.device) // group_size
    scale_full = scales.t()[:, group_idx]  # [N, K]: scales[k // group_size, n]
    nonzero = scale_full != 0
    safe = torch.where(nonzero, scale_full, torch.ones_like(scale_full))
    q = torch.round(latent / safe)
    q = torch.where(nonzero, q, torch.zeros_like(q))
    codes = (torch.clamp(q, -8, 7).to(torch.int32) + 8) & 0xF  # [N, K] in 0..15
    k_packed = (k + 1) // 2
    packed = torch.zeros((n, k_packed), dtype=torch.uint8, device=latent.device)
    packed[:, :] = codes[:, 0::2].to(torch.uint8)
    if k > 1:
        high = codes[:, 1::2].to(torch.uint8)
        packed[:, : high.shape[1]] |= high << 4
    return packed


def q4gsw_requant_meta(
    latent: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    n, k = latent.shape
    return latent.new_empty((n, (k + 1) // 2), dtype=torch.uint8)


name = "q4gsw_requant"
lib.define(f"{name}(Tensor latent, Tensor scales, int group_size) -> Tensor")
lib.impl(name, q4gsw_requant_impl, "CompositeExplicitAutograd")
lib.impl(name, q4gsw_requant_meta, "Meta")
q4gsw_requant_op = getattr(getattr(torch.ops, namespace), name)
