# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm._passes.size_adjust_input_pass import SizeAdjustInputPass
from executorch.backends.arm.operators.operator_validation_utils import (
    adjust_pooling_pad_if_needed,
)
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


_U55_MAX_POOL_STRIDE = 3
_U55_MAX_POOL_DIM = 65536
_U55_MAX_POOL_KERNEL_PRODUCT = 65536
_U55_MAX_POOL_KERNEL_WIDTH = 256


def _pair(value, fallback: tuple[int, int] | None = None) -> tuple[int, int]:
    if value is None:
        if fallback is None:
            raise ValueError("fallback is required when value is None")
        return fallback
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, Sequence):
        if len(value) == 0:
            if fallback is None:
                raise ValueError("fallback is required when value is empty")
            return fallback
        if len(value) < 2:
            raise ValueError("expected sequence pair")
        return (value[0], value[1])
    raise TypeError(f"Expected int or sequence pair, got {type(value)}")


# Keep these local to avoid importing operator support during pass construction;
# the constraints mirror pool_2d_support.dim_check/kernel_check for U55.
def _u55_dim_check(shape) -> bool:
    return all(
        not isinstance(dim, torch.SymInt) and 1 <= dim <= _U55_MAX_POOL_DIM
        for dim in shape[1:]
    )


def _u55_kernel_check(kernel: tuple[int, int]) -> bool:
    return (
        1 <= kernel[0] * kernel[1] <= _U55_MAX_POOL_KERNEL_PRODUCT
        and 1 <= kernel[1] <= _U55_MAX_POOL_KERNEL_WIDTH
    )


def can_decompose_large_stride_maxpool2d(
    kernel,
    stride,
    padding,
    dilation,
    ceil_mode,
    input_shape,
) -> bool:
    kernel_h, kernel_w = _pair(kernel)
    stride_h, stride_w = _pair(stride, (kernel_h, kernel_w))
    padding_h, padding_w = _pair(padding, (0, 0))
    dilation_h, dilation_w = _pair(dilation, (1, 1))
    height, width = input_shape[-2:]

    if (
        isinstance(height, torch.SymInt)
        or isinstance(width, torch.SymInt)
        or not _u55_kernel_check((kernel_h, kernel_w))
        or not _u55_kernel_check((1, kernel_w))
        or not _u55_kernel_check((1, kernel_h))
    ):
        return False

    post_h = adjust_pooling_pad_if_needed(
        height, kernel_h, stride_h, padding_h, ceil_mode
    )
    post_w = adjust_pooling_pad_if_needed(
        width, kernel_w, stride_w, padding_w, ceil_mode
    )
    padded_h = height + padding_h + post_h
    padded_w = width + padding_w + post_w
    output_h = padded_h // kernel_h
    output_w = padded_w // kernel_w
    padded_shape = (*input_shape[:-2], padded_h, padded_w)
    first_reduction_shape = (
        *input_shape[:-2],
        output_h * output_w * kernel_h,
        kernel_w,
    )
    second_reduction_shape = (*input_shape[:-2], output_h * output_w, kernel_h)
    output_shape = (*input_shape[:-2], output_h, output_w)

    return (
        max(stride_h, stride_w) > _U55_MAX_POOL_STRIDE
        and (kernel_h, kernel_w) == (stride_h, stride_w)
        and (dilation_h, dilation_w) == (1, 1)
        and _u55_dim_check(input_shape)
        and _u55_dim_check(padded_shape)
        and _u55_dim_check(first_reduction_shape)
        and _u55_dim_check(second_reduction_shape)
        and _u55_dim_check(output_shape)
    )


class DecomposeLargeStrideMaxPool2dForU55Pass(ArmOpTargetedPass):
    """Legalize non-overlapping max_pool2d with strides unsupported by U55.

    Materialize padding, including partial ceil-mode windows, before reducing
    each window with supported stride-one pools.

    Operator support uses the same eligibility check to keep these pools in the
    NPU partition. This pass rewrites them into supported NPU operations; it
    does not introduce CPU fallback. Pools that fail the check continue through
    the normal operator-support rules.

    Non-U55 profiles, including U85, use the normal TOSA/Vela path and do not
    need this U55 pooling-engine workaround.

    """

    _passes_required_after: Set[Type[ExportPass]] = {SizeAdjustInputPass}
    target_ops = (exir_ops.edge.aten.max_pool2d.default,)

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops or not get_context_spec().is_U55_subset:
            return super().call_operator(op, args, kwargs, meta)

        x = args[0]
        kernel = args[1]
        stride = args[2] if len(args) >= 3 and args[2] else kernel
        padding = args[3] if len(args) >= 4 else (0, 0)
        dilation = args[4] if len(args) >= 5 else (1, 1)
        ceil_mode = args[5] if len(args) >= 6 else False

        if not can_decompose_large_stride_maxpool2d(
            kernel,
            stride,
            padding,
            dilation,
            ceil_mode,
            x.data.shape,
        ):
            return super().call_operator(op, args, kwargs, meta)

        kernel_h, kernel_w = _pair(kernel)
        n, c, height, width = x.data.shape

        no_qparams_meta = meta.copy()
        no_qparams_meta.data = meta.data.copy()
        no_qparams_meta.data.pop("input_qparams", None)
        no_qparams_meta.data.pop("output_qparams", None)

        pad_h, pad_w = _pair(padding, (0, 0))
        if pad_h or pad_w or ceil_mode:
            post_h = adjust_pooling_pad_if_needed(
                height, kernel_h, kernel_h, pad_h, ceil_mode
            )
            post_w = adjust_pooling_pad_if_needed(
                width, kernel_w, kernel_w, pad_w, ceil_mode
            )
            pad_meta = meta.copy()
            pad_meta.data = meta.data.copy()
            pad_meta.data.pop("output_qparams", None)
            # RewritePadPass quantizes -inf to the input's minimum code.
            x = super().call_operator(
                exir_ops.edge.aten.constant_pad_nd.default,
                (x, [pad_w, post_w, pad_h, post_h], float("-inf")),
                {},
                pad_meta,
            )
            n, c, height, width = x.data.shape

        output_h = height // kernel_h
        output_w = width // kernel_w
        cropped_h = output_h * kernel_h
        cropped_w = output_w * kernel_w

        if cropped_h != height:
            x = super().call_operator(
                exir_ops.edge.aten.slice_copy.Tensor,
                (x, 2, 0, cropped_h),
                {},
                no_qparams_meta,
            )
        if cropped_w != width:
            x = super().call_operator(
                exir_ops.edge.aten.slice_copy.Tensor,
                (x, 3, 0, cropped_w),
                {},
                no_qparams_meta,
            )

        x = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (x, [n, c, output_h, kernel_h, output_w, kernel_w]),
            {},
            no_qparams_meta,
        )
        x = super().call_operator(
            exir_ops.edge.aten.permute_copy.default,
            (x, [0, 1, 2, 4, 3, 5]),
            {},
            no_qparams_meta,
        )
        x = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (x, [n, c, output_h * output_w * kernel_h, kernel_w]),
            {},
            no_qparams_meta,
        )
        x = super().call_operator(
            op,
            (x, (1, kernel_w), (1, 1), (0, 0), (1, 1), False),
            {},
            no_qparams_meta,
        )
        x = super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (x, [n, c, output_h * output_w, kernel_h]),
            {},
            no_qparams_meta,
        )
        x = super().call_operator(
            op,
            (x, (1, kernel_h), (1, 1), (0, 0), (1, 1), False),
            {},
            no_qparams_meta,
        )
        return super().call_operator(
            exir_ops.edge.aten.view_copy.default,
            (x, [n, c, output_h, output_w]),
            {},
            meta,
        )
