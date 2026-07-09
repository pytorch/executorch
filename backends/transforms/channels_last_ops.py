# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The ``channels_last`` operator dialect.

Operators in this dialect interpret their activation input/output as channels-last
``(N, H, W, C)`` with contiguous strides and a fixed (identity) dim-order, as
opposed to the implicit dim-order handling used elsewhere. They let layout-handling
passes (see RFC #19299) make channels-last regions explicit in the graph.

Efficiency is a non-goal: kernels are implemented as ``permute -> aten op -> permute``.
Importing this module registers the dialect.
"""

import torch
from torch.library import Library, register_fake

lib = Library("channels_last", "DEF")


def _conv(
    input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
):
    nchw = input.permute(0, 3, 1, 2)
    out = torch.ops.aten.convolution(
        nchw,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    return out.permute(0, 2, 3, 1).contiguous()


def _avg_pool2d(
    input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
):
    nchw = input.permute(0, 3, 1, 2)
    out = torch.ops.aten.avg_pool2d(
        nchw,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    return out.permute(0, 2, 3, 1).contiguous()


def _permute_copy(input, dims):
    return torch.ops.aten.permute_copy(input, dims).contiguous()


lib.define(
    "convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, "
    "int[] padding, int[] dilation, bool transposed, int[] output_padding, "
    "int groups) -> Tensor"
)
lib.impl("convolution", _conv, "CompositeExplicitAutograd")
register_fake("channels_last::convolution", _conv, lib=lib)

lib.define(
    "avg_pool2d(Tensor input, int[2] kernel_size, int[2] stride, int[2] padding, "
    "bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor"
)
lib.impl("avg_pool2d", _avg_pool2d, "CompositeExplicitAutograd")
register_fake("channels_last::avg_pool2d", _avg_pool2d, lib=lib)

lib.define("permute_copy(Tensor input, int[] dims) -> Tensor")
lib.impl("permute_copy", _permute_copy, "CompositeExplicitAutograd")
register_fake("channels_last::permute_copy", _permute_copy, lib=lib)
