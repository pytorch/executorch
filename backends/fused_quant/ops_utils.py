# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def compute_conv_out_shape(
    inp: torch.Tensor,
    weight: torch.Tensor,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> list[int]:
    """Compute output shape for NCHW convolution.

    Args:
        inp: Input tensor in NCHW format (batch, channels, *spatial)
        weight: Weight tensor in OIHW or IOHW format
        stride: Stride for each spatial dimension
        padding: Padding for each spatial dimension
        dilation: Dilation for each spatial dimension
        transposed: Whether this is a transposed convolution
        output_padding: Additional output size for transposed convolution
        groups: Number of blocked connections from inputs to outputs

    Returns:
        Output shape in NCHW format [batch, out_channels, *spatial_out]
    """
    batch_size = inp.shape[0]
    out_channels = weight.shape[1] * groups if transposed else weight.shape[0]
    spatial_dims = []
    for i in range(len(stride)):
        input_size = inp.shape[i + 2]  # Skip N, C dimensions
        kernel_size = weight.shape[i + 2]
        if transposed:
            out_size = (
                (input_size - 1) * stride[i]
                - 2 * padding[i]
                + dilation[i] * (kernel_size - 1)
                + output_padding[i]
                + 1
            )
        else:
            out_size = (
                input_size + 2 * padding[i] - dilation[i] * (kernel_size - 1) - 1
            ) // stride[i] + 1
        spatial_dims.append(out_size)

    return [batch_size, out_channels] + spatial_dims


def compute_conv_out_shape_nhwc(
    inp: torch.Tensor,
    weight: torch.Tensor,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
) -> list[int]:
    """Compute output shape for NHWC convolution.

    Args:
        inp: Input tensor in NHWC format (batch, *spatial, channels)
        weight: Weight tensor in OHWI or IHWO format
        stride: Stride for each spatial dimension
        padding: Padding for each spatial dimension
        dilation: Dilation for each spatial dimension
        transposed: Whether this is a transposed convolution
        output_padding: Additional output size for transposed convolution
        groups: Number of blocked connections from inputs to outputs

    Returns:
        Output shape in NHWC format [batch, *spatial_out, out_channels]
    """
    batch_size = inp.shape[0]
    out_channels = weight.shape[-1] * groups if transposed else weight.shape[0]
    num_spatial_dims = len(stride)
    spatial_dims = []
    for i in range(num_spatial_dims):
        input_size = inp.shape[i + 1]  # Skip N, spatial dims start at index 1
        kernel_size = weight.shape[i + 1]  # OHWI: spatial dims at indices 1..n
        if transposed:
            out_size = (
                (input_size - 1) * stride[i]
                - 2 * padding[i]
                + dilation[i] * (kernel_size - 1)
                + output_padding[i]
                + 1
            )
        else:
            out_size = (
                input_size + 2 * padding[i] - dilation[i] * (kernel_size - 1) - 1
            ) // stride[i] + 1
        spatial_dims.append(out_size)

    return [batch_size] + spatial_dims + [out_channels]
