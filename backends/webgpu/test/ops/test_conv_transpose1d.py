# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""1-D transposed `aten.convolution` module + configs for the WebGPU tests.

Imported by `cases.py` and `test_webgpu_partitioner.py`. The runtime runs these
on the transposed 2-D kernels with H = 1; `gemm_*` configs are well above the
stride == kernel GEMM path's size threshold, the others take the gather kernel.
"""

from typing import Optional

import torch


class ConvTranspose1dModule(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = True,
        activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
            bias=bias,
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.activation == "relu":
            return torch.relu(y)
        if self.activation == "hardtanh":
            return torch.nn.functional.hardtanh(y, -0.5, 0.5)
        return y


# name -> (construct kwargs, input shape (B, C, L)).
CONFIGS = {
    "stride1": ({"in_channels": 4, "out_channels": 6, "kernel_size": 3}, (1, 4, 16)),
    "stride2_padding": (
        {
            "in_channels": 4,
            "out_channels": 6,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
        },
        (1, 4, 16),
    ),
    "output_padding": (
        {
            "in_channels": 4,
            "out_channels": 6,
            "kernel_size": 5,
            "stride": 3,
            "padding": 1,
            "output_padding": 2,
        },
        (1, 4, 9),
    ),
    # output_padding >= stride is valid while it is < dilation.
    "output_padding_dilation": (
        {
            "in_channels": 4,
            "out_channels": 6,
            "kernel_size": 3,
            "stride": 1,
            "dilation": 2,
            "output_padding": 1,
        },
        (1, 4, 8),
    ),
    "groups_no_bias": (
        {
            "in_channels": 4,
            "out_channels": 6,
            "kernel_size": 3,
            "stride": 2,
            "output_padding": 1,
            "groups": 2,
            "bias": False,
        },
        (1, 4, 10),
    ),
    "dilation": (
        {
            "in_channels": 4,
            "out_channels": 4,
            "kernel_size": 3,
            "stride": 2,
            "padding": 2,
            "output_padding": 1,
            "dilation": 2,
        },
        (1, 4, 8),
    ),
    # Streaming audio decoder upsampler shape: depthwise, x16.
    "depthwise_x16": (
        {
            "in_channels": 8,
            "out_channels": 8,
            "kernel_size": 32,
            "stride": 16,
            "groups": 8,
        },
        (1, 8, 4),
    ),
    "batch2_gather": (
        {
            "in_channels": 4,
            "out_channels": 6,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
        },
        (2, 4, 16),
    ),
    "length1": (
        {"in_channels": 4, "out_channels": 6, "kernel_size": 3, "stride": 2},
        (1, 4, 1),
    ),
    "kernel1": (
        {"in_channels": 4, "out_channels": 6, "kernel_size": 1, "stride": 2},
        (1, 4, 9),
    ),
    "gemm_k4s4": (
        {"in_channels": 32, "out_channels": 64, "kernel_size": 4, "stride": 4},
        (1, 32, 2048),
    ),
    "batch2_gemm": (
        {"in_channels": 32, "out_channels": 64, "kernel_size": 4, "stride": 4},
        (2, 32, 2048),
    ),
}

# Followed by an activation FuseClampPass would fold into conv_with_clamp.
CLAMP_CONFIGS = {
    "relu": (
        {
            "in_channels": 4,
            "out_channels": 6,
            "kernel_size": 4,
            "stride": 2,
            "activation": "relu",
        },
        (1, 4, 16),
    ),
    "hardtanh": (
        {
            "in_channels": 4,
            "out_channels": 6,
            "kernel_size": 4,
            "stride": 2,
            "activation": "hardtanh",
        },
        (1, 4, 16),
    ),
}
