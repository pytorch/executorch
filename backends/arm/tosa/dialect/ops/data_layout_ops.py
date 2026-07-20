# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable

import torch

from executorch.backends.arm.constants import MAX_RANK
from executorch.backends.arm.tosa.dialect.lib import TosaValueError
from executorch.backends.arm.tosa.dialect.ops_registration import register_fake_tosa_op
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)


def _supported_data_layout_dtypes(
    allow_int16_without_extension: bool,
) -> set[torch.dtype]:
    tosa_spec = get_context_spec()
    supported_dtypes = {torch.bool}

    if tosa_spec.support_integer():
        supported_dtypes.update({torch.int8, torch.int32})
        if allow_int16_without_extension or tosa_spec.support_extension("int16"):
            supported_dtypes.add(torch.int16)
    if tosa_spec.support_float():
        supported_dtypes.update({torch.float16, torch.float32})
    if tosa_spec.support_extension("int64"):
        supported_dtypes.add(torch.int64)
    if tosa_spec.support_extension("bf16"):
        supported_dtypes.add(torch.bfloat16)
    if tosa_spec.support_extension("fp8e4m3"):
        supported_dtypes.add(torch.float8_e4m3fn)
    if tosa_spec.support_extension("fp8e5m2"):
        supported_dtypes.add(torch.float8_e5m2)

    return supported_dtypes


def _validate_data_layout_dtype(
    dtype: torch.dtype, op: str, allow_int16_without_extension: bool = True
) -> None:
    supported_dtypes = _supported_data_layout_dtypes(allow_int16_without_extension)
    if dtype not in supported_dtypes:
        raise TosaValueError(
            f"Unsupported dtype {dtype} for {op}. Supported dtypes are {supported_dtypes}",
            op=op,
        )


def _validate_data_layout_tensor(
    x: torch.Tensor, op: str, allow_int16_without_extension: bool = True
) -> None:
    _validate_data_layout_dtype(x.dtype, op, allow_int16_without_extension)


def _validate_concat_tensor(x: torch.Tensor) -> None:
    _validate_data_layout_tensor(x, "CONCAT", allow_int16_without_extension=False)


def _shape_product(shape: Iterable[int | torch.SymInt], op: str) -> int | torch.SymInt:
    result: int | torch.SymInt = 1
    for dim in shape:
        if dim < 0:
            raise TosaValueError(
                f"Negative dimension {dim} is not allowed in shape {shape}",
                op=op,
            )
        result = result * dim
    return result


@register_fake_tosa_op(
    "CONCAT(Tensor[] input1, *, int axis) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def CONCAT(inputs: list[torch.Tensor], *, axis: int) -> torch.Tensor:
    if not inputs:
        raise TosaValueError("CONCAT requires at least one input tensor", op="CONCAT")

    reference = inputs[0]
    _validate_concat_tensor(reference)

    if axis < 0 or axis >= max(1, reference.dim()):
        raise TosaValueError(
            f"CONCAT axis {axis} is out of range for rank {reference.dim()}",
            op="CONCAT",
        )

    output_shape = list(reference.shape)
    axis_sum = 0
    for tensor in inputs:
        _validate_concat_tensor(tensor)
        if tensor.dtype != reference.dtype:
            raise TosaValueError(
                "CONCAT requires matching dtypes, got "
                f"{reference.dtype} and {tensor.dtype}",
                op="CONCAT",
            )
        if tensor.dim() < 1 or tensor.dim() > MAX_RANK:
            raise TosaValueError(
                f"CONCAT input tensors must have rank between 1 and {MAX_RANK}, got {tensor.dim()}",
                op="CONCAT",
            )
        if tensor.dim() != reference.dim():
            raise TosaValueError(
                "CONCAT requires matching ranks, got "
                f"{reference.dim()} and {tensor.dim()}",
                op="CONCAT",
            )
        for dim, (lhs, rhs) in enumerate(zip(reference.shape, tensor.shape)):
            if dim != axis and lhs != rhs:
                raise TosaValueError(
                    "CONCAT requires matching non-axis dimensions, "
                    f"got {tuple(reference.shape)} and {tuple(tensor.shape)}",
                    op="CONCAT",
                )
        axis_sum = axis_sum + tensor.shape[axis]

    output_shape[axis] = axis_sum
    return torch.empty(size=output_shape, dtype=reference.dtype)


@register_fake_tosa_op(
    "PAD(Tensor input1, SymInt[] padding, *, Scalar value) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def PAD(x: torch.Tensor, padding: list[int | torch.SymInt], *, value) -> torch.Tensor:
    _validate_data_layout_dtype(x.dtype, "PAD")

    if len(padding) != 2 * len(x.shape):
        raise TosaValueError(
            f"Padding length {len(padding)} is not compatible with input rank {len(x.shape)}",
            op="PAD",
        )

    output_shape: list[int | torch.SymInt] = []
    for i, dim in enumerate(x.shape):
        pad_before = padding[i * 2]
        pad_after = padding[i * 2 + 1]
        if pad_before < 0 or pad_after < 0:
            raise TosaValueError(
                f"Expected padding values to be non-negative, got {pad_before} and {pad_after}",
                op="PAD",
            )
        output_shape.append(pad_before + dim + pad_after)

    return torch.empty(size=output_shape, dtype=x.dtype)


@register_fake_tosa_op(
    "RESHAPE(Tensor input1, SymInt[] shape) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def RESHAPE(x: torch.Tensor, shape: list[int | torch.SymInt]) -> torch.Tensor:
    _validate_data_layout_tensor(x, "RESHAPE")
    if _shape_product(x.shape, "RESHAPE") != _shape_product(shape, "RESHAPE"):
        raise TosaValueError(
            "RESHAPE requires the same number of elements, got "
            f"{tuple(x.shape)} -> {tuple(shape)}",
            op="RESHAPE",
        )
    return torch.empty(size=shape, dtype=x.dtype)


@register_fake_tosa_op(
    "REVERSE(Tensor input1, *, int axis) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def REVERSE(x: torch.Tensor, *, axis: int) -> torch.Tensor:
    _validate_data_layout_tensor(x, "REVERSE")
    if x.dim() < 1:
        raise TosaValueError("REVERSE requires rank >= 1 input", op="REVERSE")
    if axis < 0 or axis >= x.dim():
        raise TosaValueError(
            f"REVERSE axis {axis} is out of range for rank {x.dim()}",
            op="REVERSE",
        )
    return torch.empty_like(x)


@register_fake_tosa_op(
    "SLICE(Tensor input1, SymInt[] start, SymInt[] size) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def SLICE(
    x: torch.Tensor, start: list[int | torch.SymInt], size: list[int | torch.SymInt]
) -> torch.Tensor:
    input_rank = x.dim()
    if input_rank != len(start):
        raise TosaValueError(
            f"start list does not have the same rank {len(start)} as input {input_rank}",
            op="SLICE",
        )
    if len(start) != len(size):
        raise TosaValueError(
            f"size list does not have the same rank {len(size)} as start list {len(start)}",
            op="SLICE",
        )

    for i, dim_start in enumerate(start):
        if dim_start < 0 or dim_start > x.shape[i]:
            raise TosaValueError(
                f"Expected start values between [0, {x.shape[i]}] but got {dim_start}",
                op="SLICE",
            )
        dim_size = size[i]
        if dim_size <= 0 or dim_start + dim_size > x.shape[i]:
            raise TosaValueError(
                f"Expected start + size values between [0, {x.shape[i]}] but got {dim_start + dim_size}",
                op="SLICE",
            )

    _validate_data_layout_dtype(x.dtype, "SLICE")

    return torch.empty(size=size, dtype=x.dtype)


@register_fake_tosa_op(
    "TILE(Tensor input1, SymInt[] multiples) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def TILE(x: torch.Tensor, multiples: list[int | torch.SymInt]) -> torch.Tensor:
    _validate_data_layout_tensor(x, "TILE")
    if len(multiples) != x.dim():
        raise TosaValueError(
            f"TILE multiples length {len(multiples)} does not match rank {x.dim()}",
            op="TILE",
        )
    output_shape = []
    for dim, multiple in enumerate(multiples):
        if multiple <= 0:
            raise TosaValueError(
                f"TILE multiples must be positive, got {multiple} at dimension {dim}",
                op="TILE",
            )
        output_shape.append(x.shape[dim] * multiple)
    return torch.empty(size=output_shape, dtype=x.dtype)


@register_fake_tosa_op(
    "TRANSPOSE(Tensor input, int[] perms) -> Tensor",
    TosaSpecification.all_versions_and_profiles(),
)
def TRANSPOSE(x: torch.Tensor, perms: list[int]) -> torch.Tensor:
    _validate_data_layout_tensor(x, "TRANSPOSE")
    input_rank = x.dim()

    if input_rank < 1 or input_rank > MAX_RANK:
        raise TosaValueError(
            f"TRANSPOSE requires rank in [1, {MAX_RANK}], got {input_rank}",
            op="TRANSPOSE",
        )

    if len(perms) != input_rank:
        raise TosaValueError(
            f"Expected permutation rank {input_rank}, got {len(perms)}",
            op="TRANSPOSE",
        )

    seen_dims: set[int] = set()
    for dim in perms:
        if dim < 0 or dim >= input_rank or dim in seen_dims:
            raise TosaValueError(
                f"Invalid permutation {perms} for rank-{input_rank} input",
                op="TRANSPOSE",
            )
        seen_dims.add(dim)

    output_shape = [x.shape[dim] for dim in perms]
    return torch.empty(size=output_shape, dtype=x.dtype)
