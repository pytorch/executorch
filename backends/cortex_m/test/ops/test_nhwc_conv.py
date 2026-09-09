# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.cortex_m.test.ops.nhwc_test_utils import (
    int8_values,
    run_on_fvp,
)
from executorch.exir.dialects._ops import ops as exir_ops

# Direct-op coverage is temporary until CortexMTester uses explicit layout by default.


class Conv2dNhwc(torch.nn.Module):
    def __init__(self, grouped=False):
        super().__init__()
        in_channels = 4 if grouped else 3
        self.register_buffer(
            "weight", int8_values((4, 2, 3, 2 if grouped else in_channels))
        )
        self.register_buffer("bias", torch.arange(4, dtype=torch.int32) - 2)
        self.register_buffer(
            "multipliers", torch.full((4,), 1 << 30, dtype=torch.int32)
        )
        self.register_buffer("shifts", torch.full((4,), -1, dtype=torch.int32))

    def forward(self, x, scratch):
        return torch.ops.cortex_m.quantized_conv2d_nhwc.default(
            x,
            self.weight,
            self.bias,
            [2, 1],
            [1, 0],
            [1, 1],
            0,
            0,
            self.multipliers,
            self.shifts,
            -128,
            127,
            scratch,
        )


class DepthwiseConv2dNhwc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", int8_values((1, 3, 2, 4)))
        self.register_buffer("bias", torch.arange(4, dtype=torch.int32) - 2)
        self.register_buffer(
            "multipliers", torch.full((4,), 1 << 30, dtype=torch.int32)
        )
        self.register_buffer("shifts", torch.full((4,), -1, dtype=torch.int32))

    def forward(self, x, scratch):
        return torch.ops.cortex_m.quantized_depthwise_conv2d_nhwc.default(
            x,
            self.weight,
            self.bias,
            [2, 1],
            [1, 0],
            [1, 1],
            1,
            0,
            0,
            self.multipliers,
            self.shifts,
            -128,
            127,
            scratch,
        )


class TransposeConv2dNhwc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", int8_values((4, 2, 4, 2)))
        self.register_buffer("bias", torch.arange(4, dtype=torch.int32) - 2)
        self.register_buffer(
            "multipliers", torch.full((4,), 1 << 30, dtype=torch.int32)
        )
        self.register_buffer("shifts", torch.full((4,), -1, dtype=torch.int32))

    def forward(self, x, scratch, output_scratch):
        return torch.ops.cortex_m.quantized_transpose_conv2d_nhwc.default(
            x,
            self.weight,
            self.bias,
            [1, 1],
            [0, 0],
            [0, 0],
            [1, 1],
            0,
            0,
            self.multipliers,
            self.shifts,
            -128,
            127,
            scratch,
            output_scratch,
        )


def test_conv2d_nhwc_runs_on_fvp(cortex_m_target):
    run_on_fvp(
        Conv2dNhwc(),
        int8_values((1, 7, 10, 3)),
        exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default,
        cortex_m_target,
        1,
    )


def test_grouped_conv2d_nhwc_runs_on_fvp(cortex_m_target):
    run_on_fvp(
        Conv2dNhwc(grouped=True),
        int8_values((1, 7, 10, 4)),
        exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default,
        cortex_m_target,
        1,
    )


def test_depthwise_conv2d_nhwc_runs_on_fvp(cortex_m_target):
    run_on_fvp(
        DepthwiseConv2dNhwc(),
        int8_values((1, 7, 10, 4)),
        exir_ops.edge.cortex_m.quantized_depthwise_conv2d_nhwc.default,
        cortex_m_target,
        1,
    )


def test_transpose_conv2d_nhwc_runs_on_fvp(cortex_m_target):
    run_on_fvp(
        TransposeConv2dNhwc(),
        int8_values((1, 5, 6, 2)),
        exir_ops.edge.cortex_m.quantized_transpose_conv2d_nhwc.default,
        cortex_m_target,
        2,
    )
