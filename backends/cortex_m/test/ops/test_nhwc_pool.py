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


class AvgPool2dNhwc(torch.nn.Module):
    def forward(self, x, scratch):
        return torch.ops.cortex_m.quantized_avg_pool2d_nhwc.default(
            x,
            [2, 3],
            [2, 1],
            [0, 1],
            False,
            0,
            1 << 30,
            1,
            scratch,
        )


class MaxPool2dNhwc(torch.nn.Module):
    def forward(self, x):
        return torch.ops.cortex_m.quantized_max_pool2d_nhwc.default(
            x,
            [2, 3],
            [2, 1],
            [0, 1],
            [1, 1],
            False,
            0,
            0,
            -128,
            127,
        )


def test_avg_pool2d_nhwc_runs_on_fvp(cortex_m_target):
    run_on_fvp(
        AvgPool2dNhwc(),
        int8_values((1, 7, 9, 3)),
        exir_ops.edge.cortex_m.quantized_avg_pool2d_nhwc.default,
        cortex_m_target,
        1,
        atol=1,
    )


def test_max_pool2d_nhwc_runs_on_fvp(cortex_m_target):
    run_on_fvp(
        MaxPool2dNhwc(),
        int8_values((1, 7, 9, 3)),
        exir_ops.edge.cortex_m.quantized_max_pool2d_nhwc.default,
        cortex_m_target,
        0,
    )
