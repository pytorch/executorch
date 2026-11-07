# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.cortex_m.test.tester import CortexMTester, ramp_tensor


class Binary(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x, y):
        return self.op(x, y)


def _make_tester(op, shape, reverse, use_explicit_layout, target_config):
    memory_format = (
        torch.contiguous_format if use_explicit_layout else torch.channels_last
    )
    inputs = (
        ramp_tensor(-5, 5, shape).to(memory_format=memory_format),
        ramp_tensor(-1, 2, (1, shape[1], 1, 1)).to(memory_format=memory_format),
    )
    if reverse:
        inputs = inputs[::-1]
    return CortexMTester(Binary(op), inputs, target_config=target_config)


@pytest.mark.parametrize("op", [torch.add, torch.mul, torch.sub])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "use_explicit_layout", [False, True], ids=["legacy", "explicit"]
)
@pytest.mark.parametrize(
    "shape",
    [(2, 4, 3, 5), (2, 4, 1, 5), (2, 1, 3, 5), (2, 4, 1, 1)],
    ids=["ordinary", "singleton_height", "singleton_channel", "singleton_spatial"],
)
def test_dialect_channel_broadcast(
    op, reverse, use_explicit_layout, shape, cortex_m_target
):
    tester = _make_tester(op, shape, reverse, use_explicit_layout, cortex_m_target)
    kernel = "mul" if op is torch.mul else "add"
    tester.test_dialect(
        {},
        {f"executorch_exir_dialects_edge__ops_cortex_m_quantized_{kernel}_default": 1},
        qtol=1,
        use_explicit_layout=use_explicit_layout,
    )
    tester.to_executorch()


@pytest.mark.parametrize("op", [torch.add, torch.mul, torch.sub])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "use_explicit_layout", [False, True], ids=["legacy", "explicit"]
)
@pytest.mark.parametrize("shape", [(2, 4, 3, 5), (2, 4, 1, 5)])
def test_implementation_channel_broadcast(
    op, reverse, use_explicit_layout, shape, cortex_m_target
):
    tester = _make_tester(op, shape, reverse, use_explicit_layout, cortex_m_target)
    tester.test_implementation(qtol=1, use_explicit_layout=use_explicit_layout)


def test_explicit_channel_broadcast_chain(cortex_m_target):
    tester = _make_tester(
        lambda x, y: torch.relu(x + y) * y,
        (2, 4, 3, 5),
        False,
        True,
        cortex_m_target,
    )
    tester.test_dialect(
        {},
        {
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 1,
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_default": 1,
        },
        qtol=1,
        use_explicit_layout=True,
    )
    tester.to_executorch()


def test_explicit_conv_channel_broadcast(cortex_m_target):
    class ConvBroadcast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 1)

        def forward(self, x, y):
            return self.conv(x) * y

    tester = CortexMTester(
        ConvBroadcast().eval(),
        (ramp_tensor(-5, 5, (2, 3, 3, 5)), ramp_tensor(-1, 2, (1, 4, 1, 1))),
        target_config=cortex_m_target,
    )
    tester.test_dialect(
        {},
        {
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_nhwc_default": 1,
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_default": 1,
        },
        qtol=1,
        use_explicit_layout=True,
    )
    tester.to_executorch()


@pytest.mark.parametrize("op", [torch.add, torch.mul, torch.sub])
def test_explicit_spatial_broadcast_falls_back(op, cortex_m_target):
    tester = CortexMTester(
        Binary(op),
        (ramp_tensor(-5, 5, (2, 1, 3, 1)), ramp_tensor(-1, 2, (1, 4, 1, 5))),
        target_config=cortex_m_target,
    )
    tester.test_dialect(
        {},
        {f"executorch_exir_dialects_edge__ops_aten_{op.__name__}_Tensor": 1},
        use_explicit_layout=True,
    )
    tester.to_executorch()
