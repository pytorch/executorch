# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm.tosa.specification import TosaSpecification

SupportedCastMap = dict[torch.dtype, list[torch.dtype]]
SupportedCastSet = set[tuple[torch.dtype, torch.dtype]]

BOOL_TO_FP32_CAST = ((torch.bool, torch.float32),)

BASE_INT_CASTS = (
    (torch.bool, torch.int8),
    (torch.bool, torch.int16),
    (torch.bool, torch.int32),
    (torch.int8, torch.bool),
    (torch.int8, torch.int16),
    (torch.int8, torch.int32),
    (torch.int16, torch.bool),
    (torch.int16, torch.int8),
    (torch.int16, torch.int32),
    (torch.int32, torch.bool),
    (torch.int32, torch.int8),
    (torch.int32, torch.int16),
)

BASE_FP_CASTS = (
    (torch.float16, torch.float32),
    (torch.float16, torch.int8),
    (torch.float16, torch.int16),
    (torch.float16, torch.int32),
    (torch.float32, torch.bool),
    (torch.float32, torch.float16),
    (torch.float32, torch.int8),
    (torch.float32, torch.int16),
    (torch.float32, torch.int32),
    (torch.int8, torch.float16),
    (torch.int8, torch.float32),
    (torch.int16, torch.float16),
    (torch.int16, torch.float32),
    (torch.int32, torch.float16),
    (torch.int32, torch.float32),
)

BF16_CASTS = (
    (torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.int8),
    (torch.bfloat16, torch.int16),
    (torch.bfloat16, torch.int32),
    (torch.float32, torch.bfloat16),
    (torch.int8, torch.bfloat16),
    (torch.int16, torch.bfloat16),
    (torch.int32, torch.bfloat16),
)

FP8E4M3_CASTS = (
    (torch.float16, torch.float8_e4m3fn),
    (torch.float32, torch.float8_e4m3fn),
    (torch.float8_e4m3fn, torch.float16),
    (torch.float8_e4m3fn, torch.float32),
)

FP8E5M2_CASTS = (
    (torch.float16, torch.float8_e5m2),
    (torch.float32, torch.float8_e5m2),
    (torch.float8_e5m2, torch.float16),
    (torch.float8_e5m2, torch.float32),
)

BF16_FP8E4M3_CASTS = (
    (torch.bfloat16, torch.float8_e4m3fn),
    (torch.float8_e4m3fn, torch.bfloat16),
)

BF16_FP8E5M2_CASTS = (
    (torch.bfloat16, torch.float8_e5m2),
    (torch.float8_e5m2, torch.bfloat16),
)

INT64_EXTENSION_CASTS = (
    (torch.bool, torch.int64),
    (torch.int32, torch.int64),
    (torch.int64, torch.bool),
    (torch.int64, torch.int32),
)

TO_DIM_ORDER_INT_PROFILE_NOOP_CASTS = (
    (torch.bool, torch.bool),
    (torch.int8, torch.int8),
    (torch.int16, torch.int16),
    (torch.int32, torch.int32),
)

TO_DIM_ORDER_FP_PROFILE_NOOP_CASTS = (
    (torch.int8, torch.int8),
    (torch.int16, torch.int16),
    (torch.int32, torch.int32),
    (torch.float16, torch.float16),
    (torch.float32, torch.float32),
)

TO_DIM_ORDER_INT64_INPUT_CASTS = (
    (torch.int64, torch.bool),
    (torch.int64, torch.int8),
    (torch.int64, torch.int16),
    (torch.int64, torch.int32),
    (torch.int64, torch.float16),
    (torch.int64, torch.float32),
    (torch.int64, torch.bfloat16),
)


def _cast_map(casts: SupportedCastSet) -> SupportedCastMap:
    supported: SupportedCastMap = {}
    for input_dtype, output_dtype in casts:
        supported.setdefault(input_dtype, [])
        if output_dtype not in supported[input_dtype]:
            supported[input_dtype].append(output_dtype)
    return supported


def supported_tosa_casts(tosa_spec: TosaSpecification) -> SupportedCastSet:
    casts: SupportedCastSet = set()

    if tosa_spec.support_integer():
        casts.update(BASE_INT_CASTS)
    if tosa_spec.support_float():
        casts.update(BASE_FP_CASTS)
        casts.update(BOOL_TO_FP32_CAST)
    if tosa_spec.support_extension("bf16"):
        casts.update(BF16_CASTS)
    if tosa_spec.support_extension("fp8e4m3"):
        casts.update(FP8E4M3_CASTS)
        if tosa_spec.support_extension("bf16"):
            casts.update(BF16_FP8E4M3_CASTS)
    if tosa_spec.support_extension("fp8e5m2"):
        casts.update(FP8E5M2_CASTS)
        if tosa_spec.support_extension("bf16"):
            casts.update(BF16_FP8E5M2_CASTS)
    if tosa_spec.support_extension("int64"):
        casts.update(INT64_EXTENSION_CASTS)

    return casts


def supported_to_dim_order_casts(tosa_spec: TosaSpecification) -> SupportedCastMap:
    casts: SupportedCastSet = set()

    if tosa_spec.support_integer():
        casts.update(BASE_INT_CASTS)
        casts.update(TO_DIM_ORDER_INT_PROFILE_NOOP_CASTS)
        casts.update(TO_DIM_ORDER_INT64_INPUT_CASTS[:4])
    if tosa_spec.support_float():
        casts.update(BASE_FP_CASTS)
        casts.update(TO_DIM_ORDER_FP_PROFILE_NOOP_CASTS)
        casts.update(TO_DIM_ORDER_INT64_INPUT_CASTS[1:6])
    if tosa_spec.support_integer() and tosa_spec.support_float():
        casts.update(BOOL_TO_FP32_CAST)
    if tosa_spec.support_extension("bf16"):
        casts.update(BF16_CASTS)
        casts.add((torch.bfloat16, torch.bfloat16))
        casts.add((torch.int64, torch.bfloat16))
    if tosa_spec.support_extension("fp8e4m3"):
        casts.update(FP8E4M3_CASTS)
    if tosa_spec.support_extension("fp8e5m2"):
        casts.update(FP8E5M2_CASTS)
    if tosa_spec.support_extension("bf16") and tosa_spec.support_extension("fp8e4m3"):
        casts.update(BF16_FP8E4M3_CASTS)
    if tosa_spec.support_extension("bf16") and tosa_spec.support_extension("fp8e5m2"):
        casts.update(BF16_FP8E5M2_CASTS)

    return _cast_map(casts)
