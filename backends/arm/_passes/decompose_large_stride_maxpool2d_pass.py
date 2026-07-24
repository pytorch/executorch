# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch
from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm._passes.size_adjust_input_pass import SizeAdjustInputPass
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


_U55_MAX_POOL_STRIDE = 3


def can_decompose_large_stride_maxpool2d(
    kernel,
    stride,
    padding,
    dilation,
    ceil_mode,
    input_shape,
) -> bool:
    kernel_h, kernel_w = (kernel, kernel) if isinstance(kernel, int) else kernel
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    padding_h, padding_w = (
        (padding, padding) if isinstance(padding, int) else padding
    )
    dilation_h, dilation_w = (
        (dilation, dilation) if isinstance(dilation, int) else dilation
    )
    height, width = input_shape[-2:]

    return (
        not isinstance(height, torch.SymInt)
        and not isinstance(width, torch.SymInt)
        and max(stride_h, stride_w) > _U55_MAX_POOL_STRIDE
        and (kernel_h, kernel_w) == (stride_h, stride_w)
        and (padding_h, padding_w) == (0, 0)
        and (dilation_h, dilation_w) == (1, 1)
        and not ceil_mode
        and 1 <= kernel_h <= 256
        and 1 <= kernel_w <= 256
        and height >= kernel_h
        and width >= kernel_w
    )


class DecomposeLargeStrideMaxPool2dPass(ArmOpTargetedPass):
    """Legalize non-overlapping max_pool2d with strides unsupported by U55."""

    _passes_required_after: Set[Type[ExportPass]] = {SizeAdjustInputPass}
    target_ops = (exir_ops.edge.aten.max_pool2d.default,)

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops or not get_context_spec().is_U55_subset:
            return super().call_operator(op, args, kwargs, meta)

        x = args[0]
        kernel = args[1]
        stride = args[2]
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

        kernel_h, kernel_w = (
            (kernel, kernel) if isinstance(kernel, int) else kernel
        )
        n, c, height, width = x.data.shape
        if isinstance(height, torch.SymInt) or isinstance(width, torch.SymInt):
            return super().call_operator(op, args, kwargs, meta)

        output_h = height // kernel_h
        output_w = width // kernel_w
        cropped_h = output_h * kernel_h
        cropped_w = output_w * kernel_w

        no_qparams_meta = meta.copy()
        no_qparams_meta.data = meta.data.copy()
        no_qparams_meta.data.pop("input_qparams", None)
        no_qparams_meta.data.pop("output_qparams", None)

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
