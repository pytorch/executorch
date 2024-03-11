# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch.library import impl, impl_abstract

# NOTE: this is a hacky way to get around the fact that we can't use quantized_decomposed::embedding_byte in exir directly in eager model. That op can be found under exir/passes/_quant_patterns_and_replacements.py. Ideally we should consolidate these 2 versions.
# This op share the same signature and C++ kernel implementation with quantized_decomposed::embedding_byte.
quantized_lib = torch.library.Library(
    "llama_quantized", "DEF"
)  # to not be confused with torch.ops.quantized.* ops.
quantized_lib.define(
    "embedding_byte(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices) -> Tensor",
)

quantized_lib.define(
    "embedding_byte.out(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
)

quantized_lib.define(
    "embedding_byte.dtype(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices, *, ScalarType? dtype=None) -> Tensor",
)

quantized_lib.define(
    "embedding_byte.dtype_out(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
)


def embedding_byte_weight_checks(weight, weight_scales, weight_zero_points):
    assert weight.dtype in [
        torch.int8,
        torch.uint8,
    ], f"Expecting weights to be of dtype in [torch.int8, torch.uint8], but got {weight.dtype}"
    assert (
        weight.dim() == 2
    ), f"Expecting weight tensor to have dim()==2, but found {weight.dim()}"

    assert weight_scales.dtype in [
        torch.float16,
        torch.float32,
    ], f"Expecting weight_scales to be of dtype in [torch.float16, torch.float32], but got {weight_scales.dtype}"
    assert (
        weight_scales.dim() == 1 or weight_scales.dim() == 2
    ), f"Expecting weight_scales tensor to have rank 1 or 2, but found {weight_scales.dim()}"
    assert weight_scales.size(0) == weight.size(
        0
    ), f"Expecting weight and scale tensor to have same number of rows, but found {weight.size()} and {weight_scales.size()}"

    assert (
        weight_zero_points is None or weight_zero_points.dtype == weight_scales.dtype
    ), "Expecting weight_zero_points to be None or have same dtype as weight_scales"
    assert (
        weight_zero_points is None or weight_zero_points.dim() == 1
    ), f"Expecting weight_zero_points tensor to be None or have dim()==1, but found {weight_zero_points.dim()}"
    assert weight_zero_points is None or weight_zero_points.size(0) == weight.size(
        0
    ), f"Expecting weight_zero_points tensor to be None or have same number of rows as weights, but found {weight.size()} and {weight_zero_points.size()}"
    if not weight_zero_points:
        weight_zero_points = torch.zeros(weight.size(0))


# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_lib.define(
    "dequantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(
    quantized_lib,
    "dequantize_per_channel_embedding",
    "CompositeExplicitAutograd",
)
def dequantize_per_channel_embedding(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Affine per channel dequantization for the Tensor using the same quantization
    parameters for each channel/axis to map from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per channel quantized Tensor if combined with
       quantization parameter in the argument of this function (scales/zero_points/axis)

       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel

       zero_points (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel

       quant_min (int): minimum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): requested dtype for output Tensor (not used in computation,
       reserved for pattern matching)

    Returns:
       dequantized float32 Tensor
    """
    assert (
        input.dtype == dtype
    ), f"Expecting input to have dtype {dtype}, but got dtype: {input.dtype}"

    # todo: let scales be 2-d if we have group-wise quantization
    if scales.dim() == 1:
        scales = scales.view(input.shape[0], -1)

    assert (
        input.shape[1] % scales.shape[1] == 0
    ), f"currently only supporting input dim(1) as multiple of groupsize but got {input.shape[1]} with {scales.shape[1]} groups"

    print(f"input shape {input.shape}")
    print(f"scales shape {scales.shape}")
    input_shape_1 = input.shape[1]
    groups = scales.shape[1]
    group_size = (input_shape_1 + groups - 1) // groups
    if input_shape_1 % group_size != 0:
        print("padding input shape {input.shape}")
        padding = group_size - (input_shape_1 % group_size)
        input = F.pad(input, (0, padding))
        print(f"padded input shape {input.shape}    ")
    print(input.shape, groups, group_size)
    res = (
        input.to(torch.float32).view(input.shape[0], groups, group_size)
        - zero_points.view(input.shape[0], -1, 1)
    ) * scales.view(input.shape[0], groups, 1)
    res = res.view(input.shape[0], -1)[:, :input_shape_1]

    return res


@impl(quantized_lib, "dequantize_per_channel_embedding", "Meta")
def dequantize_per_channel_embedding_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    assert (
        input.dtype == dtype
    ), f"Expecting input to have dtype {dtype}, but got dtype: {input.dtype}"
    return torch.empty_like(input, dtype=torch.float32)


@impl(quantized_lib, "embedding_byte", "CompositeExplicitAutograd")
def embedding_byte_meta(
    weight,
    weight_scales,
    weight_zero_points,
    weight_quant_min,
    weight_quant_max,
    indices,
):
    embedding_byte_weight_checks(weight, weight_scales, weight_zero_points)
    weight = torch.ops.quantized_lib.dequantize_per_channel_embedding.default(
        weight,
        weight_scales,
        weight_zero_points,
        0,
        weight_quant_min,
        weight_quant_max,
        weight.dtype,
    )
    return torch.ops.aten.embedding.default(weight, indices)


@impl_abstract("llama_quantized::embedding_byte.out")
def embedding_byte_out_meta(
    weight,
    weight_scales,
    weight_zero_points,
    weight_quant_min,
    weight_quant_max,
    indices,
    out,
):
    return embedding_byte_meta(
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
    )


@impl(quantized_lib, "embedding_byte.dtype", "CompositeExplicitAutograd")
def embedding_byte_dtype_meta(
    weight,
    weight_scales,
    weight_zero_points,
    weight_quant_min,
    weight_quant_max,
    indices,
    *,
    dtype,
):
    embedding_byte_weight_checks(weight, weight_scales, weight_zero_points)
    weight = torch.ops.quantized_decomposed.dequantize_per_channel_embedding.default(
        weight,
        weight_scales,
        weight_zero_points,
        0,
        weight_quant_min,
        weight_quant_max,
        weight.dtype,
    )
    return torch.ops.aten.embedding.default(weight, indices).to(dtype)


@impl_abstract("llama_quantized::embedding_byte.dtype_out")
def embedding_byte_dtype_out_meta(
    weight,
    weight_scales,
    weight_zero_points,
    weight_quant_min,
    weight_quant_max,
    indices,
    *,
    dtype,
    out,
):
    return embedding_byte_dtype_meta(
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        dtype=dtype,
    )
