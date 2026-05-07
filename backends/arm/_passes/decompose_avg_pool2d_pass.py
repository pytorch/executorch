# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Set, Type

import torch
from executorch.backends.arm._passes.arm_pass import ArmPass
from executorch.backends.arm._passes.fuse_constant_ops_pass import (
    ComputeConstantOpsAOTPass,
)
from executorch.backends.arm.operators.operator_validation_utils import (
    adjust_pooling_pad_if_needed,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass

edge_avg_pool2d = (exir_ops.edge.aten.avg_pool2d.default,)
aten_avg_pool2d = (torch.ops.aten.avg_pool2d.default,)


def get_decomposition(op) -> tuple:
    if op in edge_avg_pool2d:
        return (
            exir_ops.edge.aten.constant_pad_nd.default,
            exir_ops.edge.aten.avg_pool2d.default,
            exir_ops.edge.aten.mul.Tensor,
        )
    if op in aten_avg_pool2d:
        return (
            torch.ops.aten.pad.default,
            torch.ops.aten.avg_pool2d.default,
            torch.ops.aten.mul.Tensor,
        )
    raise RuntimeError(f"Can't get avg_pool2d decomposition for op {op}")


def _compute_post_pad(
    size: int,
    kernel: int,
    stride: int,
    pad: int,
    ceil_mode: bool,
    divisor_override,
) -> int:

    if pad == 0:
        return pad
    if ceil_mode and divisor_override is None:
        return pad

    pad_adjust = adjust_pooling_pad_if_needed(size, kernel, stride, pad, ceil_mode)

    # Padding must always be above 0, the above adjustment may return -1
    if pad_adjust > 0:
        return pad_adjust
    return pad


def _get_avgpool_post_pad(
    h,
    w,
    kernel: tuple,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    ceil_mode,
    count_include_pad,
    divisor_override,
) -> tuple[list[Any], list[int]]:
    """Compute the post-padding configuration for avg_pool2d when pre-
    materializing explicit zero padding ahead of the pooling operation.

    Given the original spatial dimensions (h, w), pooling kernel size, stride,
    and explicit pre-padding amounts (pad_h, pad_w), this function returns the
    additional padding to apply on the right and bottom edges so that avg_pool2d
    with count_include_pad and/or divisor_override produces the equivalent
    result without built-in padding.

    """

    k_h, k_w = kernel
    post_h, post_w = (0, 0)
    new_pad_h, new_pad_w = pad_h, pad_w

    if not count_include_pad:
        return [new_pad_h, new_pad_w], [new_pad_h, new_pad_w]

    post_h = _compute_post_pad(h, k_h, stride_h, pad_h, ceil_mode, divisor_override)
    post_w = _compute_post_pad(w, k_w, stride_w, pad_w, ceil_mode, divisor_override)

    # Return our pre-padding calculation. Turn off built-in padding.
    return [pad_w, post_w, pad_h, post_h], [0, 0]


class DecomposeAvgPool2dPass(ArmPass):
    _passes_required_after: Set[Type[ExportPass]] = {ComputeConstantOpsAOTPass}

    def call_operator(self, op, args, kwargs, meta):
        if op not in (
            edge_avg_pool2d + aten_avg_pool2d
        ) or not self.allowed_to_transform(meta):
            return super().call_operator(op, args, kwargs, meta)

        pad_op, avgpool_op, mul_op = get_decomposition(op)

        x = args[0]
        kernel_h, kernel_w = args[1]
        kernel_size = kernel_h * kernel_w

        if len(args) > 2 and args[2] is not None:
            stride_h, stride_w = args[2]
        else:
            stride_h, stride_w = kernel_h, kernel_w
        pad_h, pad_w = args[3] if len(args) > 3 else (0, 0)
        ceil_mode = args[4] if len(args) > 4 else False
        count_include_pad = args[5] if len(args) > 5 else True
        divisor_override = args[6] if len(args) > 6 else None

        n, c, h, w = x.data.shape

        # Count_include_pad == False means that we use a different divisor for edge elements
        # When divisor_override is set, this will be overridden anyways.
        # It is easier to replace a constant divisor, so set count_include_pad == True
        if divisor_override is not None:
            count_include_pad = True

        pad, new_pad = _get_avgpool_post_pad(
            h,
            w,
            args[1],
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

        if count_include_pad and (pad_h > 0 or pad_w > 0):
            if op in aten_avg_pool2d:
                pad_args = (x, pad, "constant", 0.0)
            else:
                pad_args = (x, pad, 0.0)

            x = super().call_operator(
                pad_op,
                pad_args,
                {},
                meta,
                updated=True,
            )

        avgpool_args = (
            x,
            args[1],
            [stride_h, stride_w],
            new_pad,
            ceil_mode,
            False,
        )

        x = super().call_operator(avgpool_op, avgpool_args, kwargs, meta, updated=True)

        if divisor_override is not None and divisor_override != kernel_size:
            x = super().call_operator(
                mul_op,
                (x, super().call_scalar(kernel_size / divisor_override, meta)),
                {},
                meta,
                updated=True,
            )

        return x
