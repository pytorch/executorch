# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Callable, List, Optional, Tuple

import torch
from executorch.exir.dialects._ops import bind_pattern_to_op, ops as exir_ops
from executorch.exir.passes.replace_aten_with_edge_pass import (
    aten_to_edge,
    should_lower_to_edge,
)
from torch import fx
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib
from torch.library import impl, register_fake


__all__ = [
    "get_quant_patterns_and_replacements",
]


from torch import Tensor
from torch.library import custom_op


@custom_op("quant_fusion::_pack_embedding_weight", mutates_args=())
def _pack_embedding_weight(weight: Tensor, bitwidth: int) -> Tensor:
    num_embeddings, embedding_dim = weight.shape

    if bitwidth == 2:
        assert embedding_dim % 4 == 0, "embedding_dim must be divisible by 4"
        weight_range_shifted = weight.add(2).view(torch.uint8)
        weight_view = weight_range_shifted.view(num_embeddings, embedding_dim // 4, 4)
        weight_0 = weight_view[:, :, 0]
        weight_1 = weight_view[:, :, 1] << 2
        weight_2 = weight_view[:, :, 2] << 4
        weight_3 = weight_view[:, :, 3] << 6
        packed_weight = weight_0 | weight_1 | weight_2 | weight_3
        return packed_weight
    elif bitwidth == 4:
        assert embedding_dim % 2 == 0, "embedding_dim must be divisible by 2"
        weight_range_shifted = weight.add(8).view(torch.uint8)
        weight_view = weight_range_shifted.view(
            weight.shape[0], weight.shape[1] // 2, 2
        )
        weight_even = weight_view[:, :, 0] << 4
        weight_odd = weight_view[:, :, 1]
        packed_weight = weight_even | weight_odd
        return packed_weight
    elif bitwidth == 8:
        return weight

    raise RuntimeError(f"Unsupported bitwidth {bitwidth}")


# Use register_fake to add a ``FakeTensor`` kernel for the operator
@_pack_embedding_weight.register_fake
def _(weight, bit_width):
    assert bit_width in [2, 4, 8]
    num_embeddings, embedding_dim = weight.shape
    values_per_byte = 8 // bit_width
    assert embedding_dim % values_per_byte == 0
    return torch.empty(
        num_embeddings,
        embedding_dim // values_per_byte,
        dtype=torch.uint8,
        device=weight.device,
    )


# TODO: extending an existing library that is defined in OSS might be a bit
# confusing, we can investigate if it is possible to define a new library

quantized_decomposed_lib.define(
    "embedding_byte(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices) -> Tensor",
)

quantized_decomposed_lib.define(
    "embedding_byte.dtype(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices, *, ScalarType? dtype=None) -> Tensor",
)

quantized_decomposed_lib.define(
    "embedding_byte.out(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
)

quantized_decomposed_lib.define(
    "embedding_byte.dtype_out(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
)


def embedding_weight_checks(weight, weight_scales, weight_zero_points):
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
    assert weight_zero_points is None or weight_zero_points.dim() in [
        1,
        2,
    ], f"Expecting weight_zero_points tensor to be None or have dim()==1, but found {weight_zero_points.dim()}"
    assert weight_zero_points is None or weight_zero_points.size(0) == weight.size(
        0
    ), f"Expecting weight_zero_points tensor to be None or have same number of rows as weights, but found {weight.size()} and {weight_zero_points.size()}"


@impl(quantized_decomposed_lib, "embedding_byte", "CompositeExplicitAutograd")
def embedding_byte(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
) -> torch.Tensor:
    embedding_weight_checks(weight, weight_scales, weight_zero_points)
    group_size = weight.size(1) // (
        weight_scales.size(1) if weight_scales.dim() == 2 else 1
    )
    weight = torch.ops.quantized_decomposed.dequantize_per_channel_group.default(
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        weight.dtype,
        group_size,
        weight_scales.dtype,
    )
    return torch.ops.aten.embedding.default(weight, indices)


@register_fake("quantized_decomposed::embedding_byte.out")
def embedding_byte_out_meta(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    return embedding_byte(
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
    )


@impl(quantized_decomposed_lib, "embedding_byte.dtype", "CompositeExplicitAutograd")
def embedding_byte_dtype(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    embedding_weight_checks(weight, weight_scales, weight_zero_points)
    group_size = weight.size(1) // (
        weight_scales.size(1) if weight_scales.dim() == 2 else 1
    )
    weight = torch.ops.quantized_decomposed.dequantize_per_channel_group.default(
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        weight.dtype,
        group_size,
        dtype,
    )
    return torch.ops.aten.embedding.default(weight, indices)


@register_fake("quantized_decomposed::embedding_byte.dtype_out")
def embedding_byte_dtype_out_meta(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
    dtype: Optional[torch.dtype],
    out: torch.Tensor,
) -> torch.Tensor:
    return embedding_byte_dtype(
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        dtype,
    )


quantized_decomposed_lib.define(
    "embedding_2bit(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices) -> Tensor",
)

quantized_decomposed_lib.define(
    "embedding_2bit.dtype(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices, *, ScalarType? dtype=None) -> Tensor",
)

quantized_decomposed_lib.define(
    "embedding_2bit.out(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
)

quantized_decomposed_lib.define(
    "embedding_2bit.dtype_out(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
)


@impl(quantized_decomposed_lib, "embedding_2bit", "CompositeExplicitAutograd")
def embedding_2bit(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
) -> torch.Tensor:
    assert (
        weight_quant_min == -2
    ), "embedding_2bit in ExecuTorch expects weight_quant_min == -2"
    assert (
        weight_quant_max == 1
    ), "embedding_2bit in ExecuTorch expects weight_quant_max == 1"

    embedding_weight_checks(weight, weight_scales, weight_zero_points)
    group_size = (4 * weight.size(1)) // (
        weight_scales.size(1) if weight_scales.dim() == 2 else 1
    )
    weight_0 = weight & 3
    weight_1 = (weight & 12) >> 2
    weight_2 = (weight & 48) >> 4
    weight_3 = (weight & 192) >> 6
    weight_unpacked = torch.stack((weight_0, weight_1, weight_2, weight_3), dim=-1)
    weight = weight_unpacked.view(weight.shape[0], -1)
    weight = weight.view(torch.int8).add(-2)

    weight = torch.ops.quantized_decomposed.dequantize_per_channel_group.default(
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        weight.dtype,
        group_size,
        weight_scales.dtype,
    )
    return torch.ops.aten.embedding.default(weight, indices)


@register_fake("quantized_decomposed::embedding_2bit")
def _(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
):
    num_embeddings, packed_embedding_dim = weight.shape
    embedding_dim = packed_embedding_dim * 4
    embedding = torch.nn.Embedding(num_embeddings, embedding_dim, device=weight.device)
    return embedding(indices)


@register_fake("quantized_decomposed::embedding_2bit.out")
def embedding_2bit_out_meta(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    return embedding_2bit(
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
    )


@impl(quantized_decomposed_lib, "embedding_2bit.dtype", "CompositeExplicitAutograd")
def embedding_2bit_dtype(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    assert (
        weight_quant_min == -2
    ), "embedding_2bit_dtype in ExecuTorch expects weight_quant_min == -2"
    assert (
        weight_quant_max == 1
    ), "embedding_2bit_dtype in ExecuTorch expects weight_quant_max == 1"

    embedding_weight_checks(weight, weight_scales, weight_zero_points)
    group_size = (4 * weight.size(1)) // (
        weight_scales.size(1) if weight_scales.dim() == 2 else 1
    )
    weight_0 = weight & 3
    weight_1 = (weight & 12) >> 2
    weight_2 = (weight & 48) >> 4
    weight_3 = (weight & 192) >> 6
    weight_unpacked = torch.stack((weight_0, weight_1, weight_2, weight_3), dim=-1)
    weight = weight_unpacked.view(weight.shape[0], -1)
    weight = weight.view(torch.int8).add(-2)

    weight = torch.ops.quantized_decomposed.dequantize_per_channel_group.default(
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        weight.dtype,
        group_size,
        dtype,
    )
    return torch.ops.aten.embedding.default(weight, indices)


@register_fake("quantized_decomposed::embedding_2bit.dtype")
def _(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    num_embeddings, packed_embedding_dim = weight.shape
    embedding_dim = packed_embedding_dim * 4
    embedding = torch.nn.Embedding(num_embeddings, embedding_dim, device=weight.device)
    return embedding(indices).to(dtype)


@register_fake("quantized_decomposed::embedding_2bit.dtype_out")
def embedding_2bit_dtype_out_meta(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
    dtype: Optional[torch.dtype],
    out: torch.Tensor,
) -> torch.Tensor:
    return embedding_2bit_dtype(
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        dtype,
    )


quantized_decomposed_lib.define(
    "embedding_4bit(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices) -> Tensor",
)

quantized_decomposed_lib.define(
    "embedding_4bit.dtype(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices, *, ScalarType? dtype=None) -> Tensor",
)

quantized_decomposed_lib.define(
    "embedding_4bit.out(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
)

quantized_decomposed_lib.define(
    "embedding_4bit.dtype_out(Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, "
    "int weight_quant_min, int weight_quant_max, Tensor indices, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)",
)


@impl(quantized_decomposed_lib, "embedding_4bit", "CompositeExplicitAutograd")
def embedding_4bit(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
) -> torch.Tensor:
    assert (
        weight_quant_min == -8
    ), "embedding_4bit in ExecuTorch expects weight_quant_min == -8"
    assert (
        weight_quant_max == 7
    ), "embedding_4bit in ExecuTorch expects weight_quant_max == 7"

    embedding_weight_checks(weight, weight_scales, weight_zero_points)
    group_size = (2 * weight.size(1)) // (
        weight_scales.size(1) if weight_scales.dim() == 2 else 1
    )
    weight_even = weight.div(16, rounding_mode="trunc")
    weight_odd = weight.remainder(16)
    weight_unpacked = torch.stack((weight_even, weight_odd), dim=-1)
    weight = weight_unpacked.view(weight.shape[0], -1)
    weight = weight.view(torch.int8).add(-8)

    weight = torch.ops.quantized_decomposed.dequantize_per_channel_group.default(
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        weight.dtype,
        group_size,
        weight_scales.dtype,
    )
    return torch.ops.aten.embedding.default(weight, indices)


@register_fake("quantized_decomposed::embedding_4bit")
def _(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
):
    num_embeddings, packed_embedding_dim = weight.shape
    embedding_dim = packed_embedding_dim * 2
    embedding = torch.nn.Embedding(num_embeddings, embedding_dim, device=weight.device)
    return embedding(indices)


@register_fake("quantized_decomposed::embedding_4bit.out")
def embedding_4bit_out_meta(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    return embedding_4bit(
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
    )


@impl(quantized_decomposed_lib, "embedding_4bit.dtype", "CompositeExplicitAutograd")
def embedding_4bit_dtype(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    assert (
        weight_quant_min == -8
    ), "embedding_4bit_dtype in ExecuTorch expects weight_quant_min == -8"
    assert (
        weight_quant_max == 7
    ), "embedding_4bit_dtype in ExecuTorch expects weight_quant_max == 7"

    embedding_weight_checks(weight, weight_scales, weight_zero_points)
    group_size = (2 * weight.size(1)) // (
        weight_scales.size(1) if weight_scales.dim() == 2 else 1
    )
    weight_even = weight.div(16, rounding_mode="trunc")
    weight_odd = weight.remainder(16)
    weight_unpacked = torch.stack((weight_even, weight_odd), dim=-1)
    weight = weight_unpacked.view(weight.shape[0], -1)
    weight = weight.view(torch.int8).add(-8)

    weight = torch.ops.quantized_decomposed.dequantize_per_channel_group.default(
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        weight.dtype,
        group_size,
        dtype,
    )
    return torch.ops.aten.embedding.default(weight, indices)


@register_fake("quantized_decomposed::embedding_4bit.dtype")
def _(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    num_embeddings, packed_embedding_dim = weight.shape
    embedding_dim = packed_embedding_dim * 2
    embedding = torch.nn.Embedding(num_embeddings, embedding_dim, device=weight.device)
    return embedding(indices).to(dtype)


@register_fake("quantized_decomposed::embedding_4bit.dtype_out")
def embedding_4bit_dtype_out_meta(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: Optional[torch.Tensor],
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
    dtype: Optional[torch.dtype],
    out: torch.Tensor,
) -> torch.Tensor:
    return embedding_4bit_dtype(
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        indices,
        dtype,
    )


quantized_decomposed_lib.define(
    "mixed_mm(Tensor input, Tensor weight, Tensor weight_scales, Tensor? weight_zero_points) -> Tensor",
)

quantized_decomposed_lib.define(
    "mixed_linear(Tensor input, Tensor weight, Tensor weight_scales, Tensor? weight_zero_points, ScalarType? dtype=None) -> Tensor",
)

quantized_decomposed_lib.define(
    "add(Tensor a, float a_scale, int a_zero_point, int a_quant_min, int a_quant_max, Tensor b, float b_scale, int b_zero_point, int b_quant_min, int b_quant_max, float out_scale, int out_zero_point, int out_quant_min, int out_quant_max) -> Tensor qc"
)

quantized_decomposed_lib.define(
    "add.scalar(Tensor qa, float a_scale, int a_zero_point, int a_quant_min, int a_quant_max, ScalarType a_dtype, Scalar b, float out_scale, int out_zero_point, int out_quant_min, int out_quant_max, ScalarType out_dtype) -> Tensor"
)

quantized_decomposed_lib.define(
    "add_relu(Tensor a, float a_scale, int a_zero_point, int a_quant_min, int a_quant_max, Tensor b, float b_scale, int b_zero_point, int b_quant_min, int b_quant_max, float out_scale, int out_zero_point, int out_quant_min, int out_quant_max) -> Tensor qc"
)


def _trace_and_lower_to_edge_ops(f: Callable) -> fx.GraphModule:
    gm = fx.symbolic_trace(f)
    for node in gm.graph.nodes:
        if node.op == "call_function" and should_lower_to_edge(node.target):
            node.target = aten_to_edge(node.target)
    gm.recompile()
    return gm


def _sixth_input_is_scalar(match, original_graph, pattern_graph):
    """check the node that's matched to the sixth input of the pattern graph

    is a scalar number
    """
    input_idx = 0
    for node in pattern_graph.nodes:
        if node.op == "placeholder":
            if input_idx == 5:
                num_node = node
            input_idx += 1
    if not isinstance(match.nodes_map[num_node], (int, float)):
        return False
    return True


def _get_binary_op_patterns_and_replacements(
    binary_op: Callable,
    qbinary_op: Callable,
    qbinary_scalar_op: Callable,
    qbinary_relu_op: Callable,
) -> List[Tuple[Callable, Callable]]:
    @bind_pattern_to_op(quantized_decomposed_lib, qbinary_op.name())
    def binary_op_pattern(
        x,
        x_scale,
        x_zero_point,
        x_qmin,
        x_qmax,
        y,
        y_scale,
        y_zero_point,
        y_qmin,
        y_qmax,
        out_scale,
        out_zero_point,
        out_qmin,
        out_qmax,
    ):
        x = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, x_scale, x_zero_point, x_qmin, x_qmax, torch.uint8
        )
        y = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            y, y_scale, y_zero_point, y_qmin, y_qmax, torch.uint8
        )

        out = binary_op(x, y)
        out = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            out, out_scale, out_zero_point, out_qmin, out_qmax, torch.uint8
        )

        return out

    def binary_op_replacement(
        x,
        x_scale,
        x_zero_point,
        x_qmin,
        x_qmax,
        y,
        y_scale,
        y_zero_point,
        y_qmin,
        y_qmax,
        out_scale,
        out_zero_point,
        out_qmin,
        out_qmax,
    ):
        out = qbinary_op(
            x,
            x_scale,
            x_zero_point,
            x_qmin,
            x_qmax,
            y,
            y_scale,
            y_zero_point,
            y_qmin,
            y_qmax,
            out_scale,
            out_zero_point,
            out_qmin,
            out_qmax,
        )

        return out

    @bind_pattern_to_op(quantized_decomposed_lib, qbinary_scalar_op.name())
    def binary_op_scalar_1_pattern(
        x,
        x_scale,
        x_zero_point,
        x_qmin,
        x_qmax,
        num,
        out_scale,
        out_zero_point,
        out_qmin,
        out_qmax,
    ):
        x = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, x_scale, x_zero_point, x_qmin, x_qmax, torch.uint8
        )

        out = binary_op(x, num)
        out = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            out, out_scale, out_zero_point, out_qmin, out_qmax, torch.uint8
        )

        return out

    def binary_op_scalar_1_replacement(
        x,
        x_scale,
        x_zero_point,
        x_qmin,
        x_qmax,
        num,
        out_scale,
        out_zero_point,
        out_qmin,
        out_qmax,
    ):
        out = qbinary_scalar_op(
            x,
            x_scale,
            x_zero_point,
            x_qmin,
            x_qmax,
            num,
            out_scale,
            out_zero_point,
            out_qmin,
            out_qmax,
        )

        return out

    @bind_pattern_to_op(quantized_decomposed_lib, qbinary_scalar_op.name())
    def binary_op_scalar_2_pattern(
        x,
        x_scale,
        x_zero_point,
        x_qmin,
        x_qmax,
        num,
        out_scale,
        out_zero_point,
        out_qmin,
        out_qmax,
    ):
        x = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, x_scale, x_zero_point, x_qmin, x_qmax, torch.uint8
        )

        out = binary_op(num, x)
        out = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            out, out_scale, out_zero_point, out_qmin, out_qmax, torch.uint8
        )

        return out

    def binary_op_scalar_2_replacement(
        x,
        x_scale,
        x_zero_point,
        x_qmin,
        x_qmax,
        num,
        out_scale,
        out_zero_point,
        out_qmin,
        out_qmax,
    ):
        out = qbinary_scalar_op(
            x,
            x_scale,
            x_zero_point,
            x_qmin,
            x_qmax,
            num,
            out_scale,
            out_zero_point,
            out_qmin,
            out_qmax,
        )

        return out

    @bind_pattern_to_op(quantized_decomposed_lib, qbinary_relu_op.name())
    def binary_relu_op_pattern(
        x,
        x_scale,
        x_zero_point,
        x_qmin,
        x_qmax,
        y,
        y_scale,
        y_zero_point,
        y_qmin,
        y_qmax,
        out_scale,
        out_zero_point,
        out_qmin,
        out_qmax,
    ):
        x = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, x_scale, x_zero_point, x_qmin, x_qmax, torch.uint8
        )
        y = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            y, y_scale, y_zero_point, y_qmin, y_qmax, torch.uint8
        )

        out = binary_op(x, y)
        out = torch.ops.aten.relu.default(out)
        out = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            out, out_scale, out_zero_point, out_qmin, out_qmax, torch.uint8
        )

        return out

    def binary_relu_op_replacement(
        x,
        x_scale,
        x_zero_point,
        x_qmin,
        x_qmax,
        y,
        y_scale,
        y_zero_point,
        y_qmin,
        y_qmax,
        out_scale,
        out_zero_point,
        out_qmin,
        out_qmax,
    ):
        out = qbinary_relu_op(
            x,
            x_scale,
            x_zero_point,
            x_qmin,
            x_qmax,
            y,
            y_scale,
            y_zero_point,
            y_qmin,
            y_qmax,
            out_scale,
            out_zero_point,
            out_qmin,
            out_qmax,
        )

        return out

    return [
        (
            _trace_and_lower_to_edge_ops(binary_relu_op_pattern),
            _trace_and_lower_to_edge_ops(binary_relu_op_replacement),
            [],
        ),
        (
            _trace_and_lower_to_edge_ops(binary_op_pattern),
            _trace_and_lower_to_edge_ops(binary_op_replacement),
            [],
        ),
        (
            _trace_and_lower_to_edge_ops(binary_op_scalar_1_pattern),
            _trace_and_lower_to_edge_ops(binary_op_scalar_1_replacement),
            [_sixth_input_is_scalar],
        ),
        (
            _trace_and_lower_to_edge_ops(binary_op_scalar_2_pattern),
            _trace_and_lower_to_edge_ops(binary_op_scalar_2_replacement),
            [_sixth_input_is_scalar],
        ),
    ]


def _get_binary_ops_patterns_and_replacements() -> (
    List[Tuple[Callable, Callable, List[Callable]]]
):

    # TODO: replace qbinary op with the ops implemented in lean mode
    binary_op_to_qbinary_ops = {
        exir_ops.edge.aten.add.Tensor: (
            exir_ops.edge.quantized_decomposed.add.default,
            exir_ops.edge.quantized_decomposed.add.scalar,
            exir_ops.edge.quantized_decomposed.add_relu.default,
        ),
    }
    pattern_and_replacements = []
    for binary_op, (qbop, qbscalar_op, qbrelu_op) in binary_op_to_qbinary_ops.items():
        pattern_and_replacements.extend(
            _get_binary_op_patterns_and_replacements(
                binary_op, qbop, qbscalar_op, qbrelu_op
            )
        )

    return pattern_and_replacements


def _get_reshape_patterns_and_replacements() -> (
    List[Tuple[Callable, Callable, List[Callable]]]
):
    def pattern(
        x,
        arg0,
        arg1,
        x_scale,
        x_zero_point,
        x_qmin,
        x_qmax,
        out_scale,
        out_zero_point,
        out_qmin,
        out_qmax,
    ):
        x = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, x_scale, x_zero_point, x_qmin, x_qmax, torch.uint8
        )

        x = torch.ops.aten._reshape_alias_copy.default(x, arg0, arg1)
        x = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            x, out_scale, out_zero_point, out_qmin, out_qmax, torch.uint8
        )

        return x

    def replacement(
        x,
        arg0,
        arg1,
        x_scale,
        x_zero_point,
        x_qmin,
        x_qmax,
        out_scale,
        out_zero_point,
        out_qmin,
        out_qmax,
    ):

        x = torch.ops.aten._reshape_alias_copy.default(x, arg0, arg1)
        return x

    return [
        (
            _trace_and_lower_to_edge_ops(pattern),
            _trace_and_lower_to_edge_ops(replacement),
            [],
        )
    ]


def _get_slice_patterns_and_replacements() -> (
    List[Tuple[Callable, Callable, List[Callable]]]
):
    def pattern(x, dim, start, end, x_scale, x_zero_point, x_qmin, x_qmax):
        x = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, x_scale, x_zero_point, x_qmin, x_qmax, torch.uint8
        )
        x = torch.ops.aten.slice_copy.Tensor(x, dim, start, end)
        x = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            x, x_scale, x_zero_point, x_qmin, x_qmax, torch.uint8
        )
        return x

    def replacement(x, dim, start, end, x_scale, x_zero_point, x_qmin, x_qmax):
        x = torch.ops.aten.slice_copy.Tensor(x, dim, start, end)
        return x

    return [
        (
            _trace_and_lower_to_edge_ops(pattern),
            _trace_and_lower_to_edge_ops(replacement),
            [],
        )
    ]


def _get_embedding_ops_patterns_and_replacements_torchao() -> (  # noqa C901
    List[Tuple[Callable, Callable, List[Callable]]]
):
    def embedding_byte_pattern(indices, int_data, group_size, scale, zero_point):
        dq = torch.ops.torchao.dequantize_affine.default(
            int_data, [1, group_size], scale, zero_point, torch.int8, -128, 127
        )
        return torch.ops.aten.embedding.default(dq, indices)

    def embedding_byte_replacement(indices, int_data, group_size, scale, zero_point):
        zero_point_dtype_cast = torch.ops.aten.to.dtype(zero_point, scale.dtype)
        return torch.ops.quantized_decomposed.embedding_byte.default(
            int_data,
            scale,
            zero_point_dtype_cast,
            -128,
            127,
            indices,
        )

    def embedding_byte_dtype_pattern(
        indices, int_data, group_size, scale, zero_point, output_dtype
    ):
        dq = torch.ops.torchao.dequantize_affine.default(
            int_data,
            [1, group_size],
            scale,
            zero_point,
            torch.int8,
            -128,
            127,
            output_dtype,
        )
        return torch.ops.aten.embedding.default(dq, indices)

    def embedding_byte_dtype_replacement(
        indices, int_data, group_size, scale, zero_point, output_dtype
    ):
        zero_point_dtype_cast = torch.ops.aten.to.dtype(zero_point, scale.dtype)
        return torch.ops.quantized_decomposed.embedding_byte.dtype(
            int_data,
            scale,
            zero_point_dtype_cast,
            -128,
            127,
            indices,
            dtype=output_dtype,
        )

    def embedding_2bit_pattern(indices, int_data, group_size, scale, zero_point):
        dq = torch.ops.torchao.dequantize_affine.default(
            int_data, [1, group_size], scale, zero_point, torch.int8, -2, 1
        )
        return torch.ops.aten.embedding.default(dq, indices)

    def embedding_2bit_replacement(indices, int_data, group_size, scale, zero_point):
        packed_int_data = torch.ops.quant_fusion._pack_embedding_weight.default(
            int_data, 2
        )
        zero_point_dtype_cast = torch.ops.aten.to.dtype(zero_point, scale.dtype)
        return torch.ops.quantized_decomposed.embedding_2bit.default(
            packed_int_data, scale, zero_point_dtype_cast, -2, 1, indices
        )

    def embedding_2bit_dtype_pattern(
        indices, int_data, group_size, scale, zero_point, output_dtype
    ):
        dq = torch.ops.torchao.dequantize_affine.default(
            int_data,
            [1, group_size],
            scale,
            zero_point,
            torch.int8,
            -2,
            1,
            output_dtype,
        )
        return torch.ops.aten.embedding.default(dq, indices)

    def embedding_2bit_dtype_replacement(
        indices, int_data, group_size, scale, zero_point, output_dtype
    ):
        packed_int_data = torch.ops.quant_fusion._pack_embedding_weight.default(
            int_data, 2
        )
        zero_point_dtype_cast = torch.ops.aten.to.dtype(zero_point, scale.dtype)
        return torch.ops.quantized_decomposed.embedding_2bit.dtype(
            packed_int_data,
            scale,
            zero_point_dtype_cast,
            -2,
            1,
            indices,
            dtype=output_dtype,
        )

    def embedding_4bit_pattern(indices, int_data, group_size, scale, zero_point):
        dq = torch.ops.torchao.dequantize_affine.default(
            int_data, [1, group_size], scale, zero_point, torch.int8, -8, 7
        )
        return torch.ops.aten.embedding.default(dq, indices)

    def embedding_4bit_replacement(indices, int_data, group_size, scale, zero_point):
        packed_int_data = torch.ops.quant_fusion._pack_embedding_weight.default(
            int_data, 4
        )
        zero_point_dtype_cast = torch.ops.aten.to.dtype(zero_point, scale.dtype)
        return torch.ops.quantized_decomposed.embedding_4bit.default(
            packed_int_data, scale, zero_point_dtype_cast, -8, 7, indices
        )

    def embedding_4bit_dtype_pattern(
        indices, int_data, group_size, scale, zero_point, output_dtype
    ):
        dq = torch.ops.torchao.dequantize_affine.default(
            int_data,
            [1, group_size],
            scale,
            zero_point,
            torch.int8,
            -8,
            7,
            output_dtype,
        )
        return torch.ops.aten.embedding.default(dq, indices)

    def embedding_4bit_dtype_replacement(
        indices, int_data, group_size, scale, zero_point, output_dtype
    ):
        packed_int_data = torch.ops.quant_fusion._pack_embedding_weight.default(
            int_data, 4
        )
        zero_point_dtype_cast = torch.ops.aten.to.dtype(zero_point, scale.dtype)
        return torch.ops.quantized_decomposed.embedding_4bit.dtype(
            packed_int_data,
            scale,
            zero_point_dtype_cast,
            -8,
            7,
            indices,
            dtype=output_dtype,
        )

    return [
        (
            _trace_and_lower_to_edge_ops(embedding_byte_pattern),
            _trace_and_lower_to_edge_ops(embedding_byte_replacement),
            [],
        ),
        (
            _trace_and_lower_to_edge_ops(embedding_byte_dtype_pattern),
            _trace_and_lower_to_edge_ops(embedding_byte_dtype_replacement),
            [],
        ),
        (
            _trace_and_lower_to_edge_ops(embedding_2bit_pattern),
            _trace_and_lower_to_edge_ops(embedding_2bit_replacement),
            [],
        ),
        (
            _trace_and_lower_to_edge_ops(embedding_2bit_dtype_pattern),
            _trace_and_lower_to_edge_ops(embedding_2bit_dtype_replacement),
            [],
        ),
        (
            _trace_and_lower_to_edge_ops(embedding_4bit_pattern),
            _trace_and_lower_to_edge_ops(embedding_4bit_replacement),
            [],
        ),
        (
            _trace_and_lower_to_edge_ops(embedding_4bit_dtype_pattern),
            _trace_and_lower_to_edge_ops(embedding_4bit_dtype_replacement),
            [],
        ),
    ]


def _get_embedding_ops_patterns_and_replacements() -> (
    List[Tuple[Callable, Callable, List[Callable]]]
):
    def get_pattern_and_replacement():
        @bind_pattern_to_op(quantized_decomposed_lib, "embedding_byte")
        def pattern(
            weight,
            weight_scales,
            weight_zero_points,
            weight_quant_min,
            weight_quant_max,
            indicies,
        ):
            weight = torch.ops.quantized_decomposed.dequantize_per_channel.default(
                weight,
                weight_scales,
                weight_zero_points,
                0,
                weight_quant_min,
                weight_quant_max,
                torch.uint8,
            )
            out = torch.ops.aten.embedding.default(weight, indicies)
            return out

        def replacement(
            weight,
            weight_scales,
            weight_zero_points,
            weight_quant_min,
            weight_quant_max,
            indicies,
        ):
            out = torch.ops.quantized_decomposed.embedding_byte.default(
                weight,
                weight_scales,
                weight_zero_points,
                weight_quant_min,
                weight_quant_max,
                indicies,
            )
            return out

        @bind_pattern_to_op(quantized_decomposed_lib, "embedding_byte")
        def pattern_groupwise(
            weight,
            weight_scales,
            weight_zero_points,
            weight_quant_min,
            weight_quant_max,
            indices,
            group_size,
        ):
            weight = (
                torch.ops.quantized_decomposed.dequantize_per_channel_group.default(
                    weight,
                    weight_scales,
                    weight_zero_points,
                    weight_quant_min,
                    weight_quant_max,
                    weight.dtype,
                    group_size,
                    weight_scales.dtype,
                )
            )
            out = torch.ops.aten.embedding.default(weight, indices)
            return out

        def replacement_groupwise(
            weight,
            weight_scales,
            weight_zero_points,
            weight_quant_min,
            weight_quant_max,
            indices,
            group_size,
        ):
            out = torch.ops.quantized_decomposed.embedding_byte.default(
                weight,
                weight_scales,
                weight_zero_points,
                weight_quant_min,
                weight_quant_max,
                indices,
            )
            return out

        @bind_pattern_to_op(quantized_decomposed_lib, "embedding_byte")
        def pattern_with_padding_idx(
            weight,
            weight_scales,
            weight_zero_points,
            weight_quant_min,
            weight_quant_max,
            indicies,
            padding_idx,
        ):
            weight = torch.ops.quantized_decomposed.dequantize_per_channel.default(
                weight,
                weight_scales,
                weight_zero_points,
                0,
                weight_quant_min,
                weight_quant_max,
                torch.uint8,
            )
            out = torch.ops.aten.embedding.default(weight, indicies, padding_idx)
            return out

        def replacement_with_padding_idx(
            weight,
            weight_scales,
            weight_zero_points,
            weight_quant_min,
            weight_quant_max,
            indicies,
            _,  # padding_idx only matters for training and not when running op for inference
        ):
            out = torch.ops.quantized_decomposed.embedding_byte.default(
                weight,
                weight_scales,
                weight_zero_points,
                weight_quant_min,
                weight_quant_max,
                indicies,
            )
            return out

        @bind_pattern_to_op(quantized_decomposed_lib, "embedding_byte")
        def pattern_with_padding_idx_groupwise(
            weight,
            weight_scales,
            weight_zero_points,
            weight_quant_min,
            weight_quant_max,
            indices,
            group_size,
            padding_idx,
        ):
            weight = (
                torch.ops.quantized_decomposed.dequantize_per_channel_group.default(
                    weight,
                    weight_scales,
                    weight_zero_points,
                    weight_quant_min,
                    weight_quant_max,
                    weight.dtype,
                    group_size,
                    weight_scales.dtype,
                )
            )
            out = torch.ops.aten.embedding.default(weight, indices, padding_idx)
            return out

        def replacement_with_padding_idx_groupwise(
            weight,
            weight_scales,
            weight_zero_points,
            weight_quant_min,
            weight_quant_max,
            indices,
            group_size,
            _,  # padding_idx only matters for training and not when running op for inference
        ):
            out = torch.ops.quantized_decomposed.embedding_byte.default(
                weight,
                weight_scales,
                weight_zero_points,
                weight_quant_min,
                weight_quant_max,
                indices,
            )
            return out

        @bind_pattern_to_op(quantized_decomposed_lib, "embedding_byte.dtype")
        def pattern_with_dtype_groupwise(
            weight,
            weight_scales,
            weight_zero_points,
            weight_quant_min,
            weight_quant_max,
            indices,
            group_size,
            dtype,
        ):
            weight = (
                torch.ops.quantized_decomposed.dequantize_per_channel_group.default(
                    weight,
                    weight_scales,
                    weight_zero_points,
                    weight_quant_min,
                    weight_quant_max,
                    weight.dtype,
                    group_size,
                    dtype,
                )
            )
            out = torch.ops.aten.embedding.default(weight, indices)
            return out

        def replacement_with_dtype_groupwise(
            weight,
            weight_scales,
            weight_zero_points,
            weight_quant_min,
            weight_quant_max,
            indices,
            group_size,
            dtype,
        ):
            out = torch.ops.quantized_decomposed.embedding_byte.dtype(
                weight,
                weight_scales,
                weight_zero_points,
                weight_quant_min,
                weight_quant_max,
                indices,
                dtype=dtype,
            )
            return out

        return [
            (
                _trace_and_lower_to_edge_ops(pattern),
                _trace_and_lower_to_edge_ops(replacement),
                [],
            ),
            (
                _trace_and_lower_to_edge_ops(pattern_groupwise),
                _trace_and_lower_to_edge_ops(replacement_groupwise),
                [],
            ),
            (
                _trace_and_lower_to_edge_ops(pattern_with_padding_idx),
                _trace_and_lower_to_edge_ops(replacement_with_padding_idx),
                [],
            ),
            (
                _trace_and_lower_to_edge_ops(pattern_with_padding_idx_groupwise),
                _trace_and_lower_to_edge_ops(replacement_with_padding_idx_groupwise),
                [],
            ),
            (
                _trace_and_lower_to_edge_ops(pattern_with_dtype_groupwise),
                _trace_and_lower_to_edge_ops(replacement_with_dtype_groupwise),
                [],
            ),
        ]

    patterns_and_replacements = []
    patterns_and_replacements.extend(
        get_pattern_and_replacement(),
    )
    return patterns_and_replacements


"""
def _get_fixed_qparams_ops_patterns_and_replacements() -> List[Tuple[Callable, Callable, List[Callable]]]:
    fixed_qparams_op_to_qop = {
        torch.ops.aten.softmax: (torch.ops.quantized_decomposed.softmax, 1.0 / 256.0, 0)
    }
    def get_pattern_and_replacement(fixed_qparams_op, fixed_scale, fixed_zero_point):
        def pattern(x, x_scale, x_zero_point, x_qmin, x_qmax):
            x = torch.ops.quantized_decomposed.dequantize_per_tensor.default(x, x_scale, x_zero_point, x_qmin, x_qmax, torch.uint8)
            x = fixed_qparams_op(x)
            x = torch.ops.quantized_decomposed.dequantize_per_tensor.default(x, fixed_scale, fixed_zero_point, 0, 255, torch.uint8)
            return x

        def replacement(x, x_scale, x_zero_point, x_qmin, x_qmax):
            x = fixed_qparams_qop(x, x_scale, x_zero_point, x_qmin, x_qmax, torch.uint8)
            return x

n        return [(pattern, replacement, [])]

    patterns_and_replacements = []
    for op, (qop, fixed_scale, fixed_zero_point) in fixed_qparams_op_to_qop.items():
        patterns_and_replacements.extend(
            get_pattern_and_replacement(op, qop, fixed_scale, fixed_zero_point)
        )
"""


def get_quant_patterns_and_replacements() -> (
    List[Tuple[Callable, Callable, List[Callable]]]
):

    return copy.copy(
        [
            *_get_binary_ops_patterns_and_replacements(),
            # TODO: enable following after the corresponding ops are implemented
            *_get_reshape_patterns_and_replacements(),
            *_get_slice_patterns_and_replacements(),
            # *_get_fixed_qparams_ops_patterns_and_replacements(),
            *_get_embedding_ops_patterns_and_replacements(),
            *_get_embedding_ops_patterns_and_replacements_torchao(),
        ]
    )
