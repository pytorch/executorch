# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.cortex_m.test.tester import CortexMTester, ramp_tensor
from executorch.backends.test.harness.stages import StageType


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
    if use_explicit_layout:
        kernel += "_nhwc"
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
@pytest.mark.parametrize(
    "shape", [(2, 4, 3, 5), (2, 4, 1, 5), (2, 1, 3, 5), (2, 4, 1, 1)]
)
def test_implementation_channel_broadcast(
    op, reverse, use_explicit_layout, shape, cortex_m_target
):
    tester = _make_tester(op, shape, reverse, use_explicit_layout, cortex_m_target)
    tester.test_implementation(qtol=1, use_explicit_layout=use_explicit_layout)


@pytest.mark.parametrize(
    "op",
    [
        lambda x, y: torch.ops.cortex_m.quantized_add.default(
            x, 0, 1 << 30, 0, y, 0, 1 << 30, -1, 0, 1 << 30, -19, -128, 127
        ),
        lambda x, y: torch.ops.cortex_m.quantized_mul.default(
            x, 0, y, 0, 0, 1 << 30, -3
        ),
    ],
    ids=["add", "mul"],
)
@pytest.mark.parametrize(
    "canonical", [True, False], ids=["canonical", "singleton_stride"]
)
def test_implementation_legacy_channel_broadcast_singleton_strides(
    op, canonical, cortex_m_target
):
    x = torch.arange(-12, 12, dtype=torch.int8).reshape(2, 3, 4, 1).transpose(1, 2)
    if canonical:
        x = torch.empty(
            x.shape, dtype=x.dtype, memory_format=torch.channels_last
        ).copy_(x)
    assert x.is_contiguous(memory_format=torch.channels_last)
    inputs = (x, torch.tensor([-16, -7, 9, 24], dtype=torch.int8).reshape(1, 4, 1, 1))
    # Quantizing float inputs would normalize the singleton strides.
    (
        CortexMTester(Binary(op), inputs, target_config=cortex_m_target)
        .export()
        .to_edge()
        .run_passes()
        .to_executorch()
        .serialize()
        .run_method_and_compare_outputs(inputs=inputs, atol=0, rtol=0)
    )


@pytest.mark.parametrize("op", [torch.add, torch.mul, torch.sub])
@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_channel_broadcast_rejects_implicit_layout(
    op, reverse, cortex_m_target
):
    tester = _make_tester(op, (2, 4, 3, 5), reverse, False, cortex_m_target)
    with pytest.raises(Exception) as caught:
        tester.test_dialect({}, {}, use_explicit_layout=True, compare_outputs=False)

    error = caught.value
    while error.__cause__ is not None:
        error = error.__cause__
    assert "contiguous NHWC" in str(error)


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
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_nhwc_default": 1,
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_nhwc_default": 1,
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
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_packed_nhwc_default": 1,
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_nhwc_default": 1,
        },
        qtol=1,
        use_explicit_layout=True,
    )
    tester.to_executorch()


def test_explicit_broadcast_branch_keeps_only_boundary_copies(cortex_m_target):
    class ConvPoolBroadcast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 1)

        def forward(self, x):
            x = self.conv(x)
            return x * torch.nn.functional.avg_pool2d(x, (3, 5))

    tester = CortexMTester(
        ConvPoolBroadcast().eval(),
        (ramp_tensor(-5, 5, (1, 3, 3, 5)),),
        target_config=cortex_m_target,
    )
    tester.test_dialect(
        {},
        {
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_packed_nhwc_default": 1,
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_avg_pool2d_nhwc_default": 1,
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_nhwc_default": 1,
            "executorch_exir_dialects_edge__ops_cortex_m_transpose_default": 2,
        },
        qtol=1,
        use_explicit_layout=True,
    )
    tester.to_executorch()


@pytest.mark.parametrize("op", [torch.add, torch.mul, torch.sub])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shape", [(0, 4, 3, 5), (2, 4, 0, 5)])
def test_explicit_empty_channel_broadcast_falls_back(op, reverse, shape):
    inputs = (torch.empty(shape), torch.arange(4).float().reshape(1, 4, 1, 1))
    if reverse:
        inputs = inputs[::-1]
    tester = CortexMTester(Binary(op), inputs)
    kernel = "mul" if op is torch.mul else "add"
    tester.test_dialect(
        {},
        {
            f"executorch_exir_dialects_edge__ops_aten_{op.__name__}_Tensor": 1,
            f"executorch_exir_dialects_edge__ops_cortex_m_quantized_{kernel}_default": 0,
            f"executorch_exir_dialects_edge__ops_cortex_m_quantized_{kernel}_nhwc_default": 0,
        },
        use_explicit_layout=True,
        compare_outputs=False,
    )
    tester.to_executorch()
    program = tester.get_artifact(StageType.TO_EXECUTORCH).executorch_program
    plan = program.execution_plan[0]
    assert [plan.values[index].val.sizes for index in plan.outputs] == [list(shape)]


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


@pytest.mark.parametrize("op", [torch.add, torch.mul, torch.sub])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "use_explicit_layout", [False, True], ids=["legacy", "explicit"]
)
def test_integer_channel_broadcast_falls_back(
    op, reverse, use_explicit_layout, cortex_m_target
):
    class IntegerChannelBroadcast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "channel",
                torch.tensor([-2, -1, 1, 2], dtype=torch.int8).reshape(1, 4, 1, 1),
            )

        def forward(self, x):
            return op(self.channel, x) if reverse else op(x, self.channel)

    x = ramp_tensor(-5, 5, (2, 4, 3, 5))
    if not use_explicit_layout:
        x = x.to(memory_format=torch.channels_last)
    tester = CortexMTester(
        IntegerChannelBroadcast(), (x,), target_config=cortex_m_target
    )
    kernel = "mul" if op is torch.mul else "add"
    tester.test_dialect(
        {},
        {
            f"executorch_exir_dialects_edge__ops_aten_{op.__name__}_Tensor": 1,
            f"executorch_exir_dialects_edge__ops_cortex_m_quantized_{kernel}_default": 0,
            f"executorch_exir_dialects_edge__ops_cortex_m_quantized_{kernel}_nhwc_default": 0,
        },
        use_explicit_layout=use_explicit_layout,
    )
    tester.to_executorch()
    assert tester.get_artifact(StageType.TO_EXECUTORCH).buffer
