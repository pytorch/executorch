# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest

import torch
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig

from executorch.backends.cortex_m.test.tester import CortexMTester
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as exir_ops


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=4,
        groups=1,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )

    def forward(self, x):
        return self.conv(x)


class Conv1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 4, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class TwoConv2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 5, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class ConvPoolConv(torch.nn.Module):
    def __init__(self, pool):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.pool = pool
        self.conv2 = torch.nn.Conv2d(4, 5, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.pool(self.conv1(x)))


class ConvTranspose2d(torch.nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=True,
    ):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class ConvPadConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 5, 3, padding=1)

    def forward(self, x):
        return self.conv2(torch.nn.functional.pad(self.conv1(x), (1, 1, 1, 1)))


class ConvBiasConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 5, 3, padding=1)
        self.bias = torch.nn.Parameter(torch.randn(1, 4, 1, 1))

    def forward(self, x):
        return self.conv2(torch.relu(self.conv1(x)) + self.bias)


class ConvBiasBranched(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 5, 3, padding=1)
        self.bias = torch.nn.Parameter(torch.randn(1, 4, 1, 1))

    def forward(self, x):
        biased = self.conv1(x) + self.bias
        return self.conv2(biased), biased


class ConvBiasOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.bias = torch.nn.Parameter(torch.randn(1, 4, 1, 1))

    def forward(self, x):
        return self.conv(x) + self.bias


class ConvMulConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 5, 3, padding=1)
        self.scale = torch.nn.Parameter(torch.randn(1, 4, 1, 1))

    def forward(self, x):
        return self.conv2(self.conv1(x) * self.scale)


class ConvPoolBiasConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.pool = torch.nn.AvgPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(4, 5, 3, padding=1)
        self.bias = torch.nn.Parameter(torch.randn(1, 4, 1, 1))

    def forward(self, x):
        return self.conv2(self.pool(self.conv1(x)) + self.bias)


class ConvForkAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.branch1 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.branch2 = torch.nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        stem = self.stem(x)
        return self.branch1(stem) + self.branch2(stem)


class ConvSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x):
        return torch.softmax(self.conv(x), dim=-1)


def _lower(module, inputs):
    tester = CortexMTester(module, inputs, use_explicit_layout=True)
    tester.quantize().export().to_edge()

    edge_program = tester.get_artifact(StageType.TO_EDGE).exported_program()
    assert all(
        getattr(node.target, "namespace", None) != "dim_order_ops"
        for node in edge_program.graph.nodes
        if node.op == "call_function"
    )

    tester.run_passes()
    tester.run_method_and_compare_outputs(inputs=inputs, qtol=2)
    return tester.get_artifact(StageType.RUN_PASSES).exported_program()


def _lower_legacy(module, inputs):
    tester = CortexMTester(module, inputs)
    tester.quantize().export().to_edge().run_passes()
    return tester.get_artifact(StageType.RUN_PASSES).exported_program()


def _run_explicit_layout_on_fvp(module, inputs, target, qtol=2):
    tester = CortexMTester(module, inputs, use_explicit_layout=True)
    tester.quantize().export().to_edge().run_passes()
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    assert _count(program, target) == 1

    tester.to_executorch().serialize()
    tester.run_method_and_compare_outputs(inputs=inputs, qtol=qtol)


def _count(exported_program, target):
    return sum(
        node.op == "call_function" and node.target == target
        for node in exported_program.graph.nodes
    )


def _planned_buffer_sizes(
    module,
    inputs,
    use_explicit_layout,
    target_config,
    expected_ops,
):
    module = copy.deepcopy(module).eval()
    inputs = tuple(value.clone() for value in inputs)
    if not use_explicit_layout:
        # Legacy kernels rely on channels-last capture for zero-copy CMSIS input.
        module.to(memory_format=torch.channels_last)
        inputs = tuple(
            value.to(memory_format=torch.channels_last) if value.dim() == 4 else value
            for value in inputs
        )

    tester = CortexMTester(
        module,
        inputs,
        target_config=target_config,
        use_explicit_layout=use_explicit_layout,
    )
    tester.quantize().export().to_edge().run_passes()
    exported_program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    for target, count in expected_ops.items():
        assert _count(exported_program, target) == count

    tester.to_executorch()
    program = tester.get_artifact(StageType.TO_EXECUTORCH).executorch_program
    return tuple(program.execution_plan[0].non_const_buffer_sizes)


def test_conv2d_uses_explicit_nhwc_operator():
    x = torch.randn(1, 3, 8, 8)
    program = _lower(Conv2d(), (x,))

    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 1
    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d.default) == 0
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 2
    assert program.module()(x).shape == torch.Size([1, 4, 8, 8])


def test_conv1d_reuses_explicit_conv2d_region():
    x = torch.randn(1, 2, 8)
    tester = CortexMTester(Conv1d(), (x,), use_explicit_layout=True)
    tester.quantize().export().to_edge().run_passes()
    tester.run_method_and_compare_outputs(inputs=(x,), qtol=1)
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()

    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 1
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 2
    assert program.module()(x).shape == torch.Size([1, 4, 8])


def test_legacy_mode_does_not_select_explicit_layout():
    x = torch.randn(1, 3, 8, 8)
    program = _lower_legacy(Conv2d(), (x,))

    assert _count(program, exir_ops.edge.aten.convolution.default) == 1
    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d.default) == 0
    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 0


def test_depthwise_conv2d_uses_explicit_nhwc_operator():
    x = torch.randn(1, 4, 8, 8)
    program = _lower(Conv2d(4, 4, groups=4), (x,))

    assert (
        _count(
            program,
            exir_ops.edge.cortex_m.quantized_depthwise_conv2d_nhwc.default,
        )
        == 1
    )
    assert (
        _count(program, exir_ops.edge.cortex_m.quantized_depthwise_conv2d.default) == 0
    )
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 2
    assert program.module()(x).shape == torch.Size([1, 4, 8, 8])


def test_grouped_conv2d_uses_explicit_nhwc_operator():
    # groups=2 is neither dense nor depthwise, so it must reach the regular
    # NHWC convolution rather than the depthwise one.
    x = torch.randn(1, 4, 8, 8)
    program = _lower(Conv2d(4, 4, groups=2), (x,))

    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 1
    assert (
        _count(
            program,
            exir_ops.edge.cortex_m.quantized_depthwise_conv2d_nhwc.default,
        )
        == 0
    )
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 2
    assert program.module()(x).shape == torch.Size([1, 4, 8, 8])


def test_adjacent_convolutions_eliminate_internal_copies():
    x = torch.randn(1, 3, 8, 8)
    program = _lower(TwoConv2d(), (x,))

    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 2
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 2


def test_avg_pool2d_joins_explicit_layout_region():
    x = torch.randn(1, 3, 8, 8)
    program = _lower(ConvPoolConv(torch.nn.AvgPool2d(2, 2)), (x,))

    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 2
    assert (
        _count(program, exir_ops.edge.cortex_m.quantized_avg_pool2d_nhwc.default) == 1
    )
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 2


def test_max_pool2d_joins_explicit_layout_region():
    x = torch.randn(1, 3, 8, 8)
    program = _lower(ConvPoolConv(torch.nn.MaxPool2d(2, 2)), (x,))

    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 2
    assert (
        _count(program, exir_ops.edge.cortex_m.quantized_max_pool2d_nhwc.default) == 1
    )
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 2


def test_transpose_conv2d_uses_explicit_nhwc_operator():
    x = torch.randn(1, 3, 5, 5)
    program = _lower(ConvTranspose2d(), (x,))

    assert (
        _count(
            program,
            exir_ops.edge.cortex_m.quantized_transpose_conv2d_nhwc.default,
        )
        == 1
    )
    assert (
        _count(program, exir_ops.edge.cortex_m.quantized_transpose_conv2d.default) == 0
    )
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 2
    assert program.module()(x).shape == torch.Size([1, 4, 9, 9])


def test_pad_is_remapped_inside_explicit_layout_region():
    x = torch.randn(1, 3, 8, 8)
    program = _lower(ConvPadConv(), (x,))
    [pad] = [
        node
        for node in program.graph.nodes
        if node.target == exir_ops.edge.cortex_m.pad_contiguous.default
    ]

    assert pad.args[1:3] == ([0, 1, 1, 0], [0, 1, 1, 0])
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 2


def test_softmax_stays_outside_explicit_layout_region():
    x = torch.randn(1, 3, 8, 8)
    program = _lower(ConvSoftmax(), (x,))
    [softmax] = [
        node
        for node in program.graph.nodes
        if node.target == exir_ops.edge.cortex_m.softmax.default
    ]

    assert softmax.args[1] in (-1, 3)
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 2


def test_channel_bias_is_quantized_inside_explicit_layout_region():
    x = torch.randn(1, 3, 8, 8)
    program = _lower(ConvBiasConv(), (x,))
    [add] = [
        node
        for node in program.graph.nodes
        if node.target == exir_ops.edge.cortex_m.quantized_add.default
    ]

    assert add.args[4].meta["val"].shape == torch.Size([1, 1, 1, 4])
    assert _count(program, exir_ops.edge.aten.add.Tensor) == 0
    assert _count(program, exir_ops.edge.cortex_m.quantized_add.default) == 1
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 2


def test_branched_channel_bias_stays_quantized():
    x = torch.randn(1, 3, 8, 8)
    program = _lower(ConvBiasBranched(), (x,))

    assert _count(program, exir_ops.edge.cortex_m.quantized_add.default) == 1
    assert _count(program, exir_ops.edge.aten.add.Tensor) == 0
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 3


def test_unanchored_channel_bias_falls_back_without_failing():
    x = torch.randn(1, 3, 8, 8)
    program = _lower(ConvBiasOutput(), (x,))

    assert _count(program, exir_ops.edge.cortex_m.quantized_add.default) == 0
    assert _count(program, exir_ops.edge.aten.add.Tensor) == 1


def test_channel_mul_is_quantized_inside_explicit_layout_region():
    x = torch.randn(1, 3, 8, 8)
    program = _lower(ConvMulConv(), (x,))
    [mul] = [
        node
        for node in program.graph.nodes
        if node.target == exir_ops.edge.cortex_m.quantized_mul.default
    ]

    assert mul.args[2].meta["val"].shape == torch.Size([1, 1, 1, 4])
    assert _count(program, exir_ops.edge.aten.mul.Tensor) == 0
    assert _count(program, exir_ops.edge.cortex_m.quantized_mul.default) == 1
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 2


def test_pool_anchor_allows_quantized_channel_broadcast():
    x = torch.randn(1, 3, 8, 8)
    program = _lower(ConvPoolBiasConv(), (x,))

    assert _count(program, exir_ops.edge.cortex_m.quantized_add.default) == 1
    assert (
        _count(program, exir_ops.edge.cortex_m.quantized_avg_pool2d_nhwc.default) == 1
    )
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 2


def test_explicit_layout_does_not_increase_planned_memory_for_float_qdq():
    torch.manual_seed(0)
    m33 = CortexMTargetConfig(cpu=CortexM.M33)
    m55 = CortexMTargetConfig(cpu=CortexM.M55)
    cases = (
        (
            Conv2d(),
            (torch.randn(1, 3, 8, 8),),
            m55,
            {exir_ops.edge.cortex_m.quantized_conv2d.default: 1},
            {exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default: 1},
        ),
        (
            TwoConv2d(),
            (torch.randn(1, 3, 8, 8),),
            m55,
            {exir_ops.edge.cortex_m.quantized_conv2d.default: 2},
            {exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default: 2},
        ),
        (
            Conv2d(4, 4, groups=4),
            (torch.randn(1, 4, 8, 8),),
            m55,
            {exir_ops.edge.cortex_m.quantized_depthwise_conv2d.default: 1},
            {exir_ops.edge.cortex_m.quantized_depthwise_conv2d_nhwc.default: 1},
        ),
        (
            ConvTranspose2d(),
            (torch.randn(1, 3, 5, 5),),
            m55,
            {exir_ops.edge.cortex_m.quantized_transpose_conv2d.default: 1},
            {exir_ops.edge.cortex_m.quantized_transpose_conv2d_nhwc.default: 1},
        ),
        (
            ConvPoolConv(torch.nn.AvgPool2d(2, 2)),
            (torch.randn(1, 3, 8, 8),),
            m55,
            {
                exir_ops.edge.cortex_m.quantized_conv2d.default: 2,
                exir_ops.edge.cortex_m.quantized_avg_pool2d.default: 1,
            },
            {
                exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default: 2,
                exir_ops.edge.cortex_m.quantized_avg_pool2d_nhwc.default: 1,
            },
        ),
        (
            ConvPoolConv(torch.nn.MaxPool2d(2, 2)),
            (torch.randn(1, 3, 8, 8),),
            m55,
            {
                exir_ops.edge.cortex_m.quantized_conv2d.default: 2,
                exir_ops.edge.cortex_m.quantized_max_pool2d.default: 1,
            },
            {
                exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default: 2,
                exir_ops.edge.cortex_m.quantized_max_pool2d_nhwc.default: 1,
            },
        ),
        (
            ConvForkAdd(),
            (torch.randn(1, 3, 8, 8),),
            m55,
            {exir_ops.edge.cortex_m.quantized_conv2d.default: 3},
            {exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default: 3},
        ),
        (
            ConvBiasConv(),
            (torch.randn(1, 3, 8, 8),),
            m55,
            {
                exir_ops.edge.cortex_m.quantized_conv2d.default: 2,
                exir_ops.edge.cortex_m.quantized_add.default: 1,
            },
            {
                exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default: 2,
                exir_ops.edge.cortex_m.quantized_add.default: 1,
            },
        ),
        (
            ConvBiasBranched(),
            (torch.randn(1, 3, 8, 8),),
            m55,
            {
                exir_ops.edge.cortex_m.quantized_conv2d.default: 2,
                exir_ops.edge.cortex_m.quantized_add.default: 1,
            },
            {
                exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default: 2,
                exir_ops.edge.cortex_m.quantized_add.default: 1,
            },
        ),
        (
            ConvMulConv(),
            (torch.randn(1, 3, 8, 8),),
            m55,
            {
                exir_ops.edge.cortex_m.quantized_conv2d.default: 2,
                exir_ops.edge.cortex_m.quantized_mul.default: 1,
            },
            {
                exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default: 2,
                exir_ops.edge.cortex_m.quantized_mul.default: 1,
            },
        ),
        (
            ConvPadConv(),
            (torch.randn(1, 3, 8, 8),),
            m55,
            {
                exir_ops.edge.cortex_m.quantized_conv2d.default: 2,
                exir_ops.edge.cortex_m.pad.default: 1,
            },
            {
                exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default: 2,
                exir_ops.edge.cortex_m.pad_contiguous.default: 1,
            },
        ),
        (
            Conv2d(),
            (torch.randn(1, 3, 8, 8),),
            m33,
            {exir_ops.edge.cortex_m.quantized_conv2d.default: 1},
            {exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default: 1},
        ),
        (
            ConvBiasConv(),
            (torch.randn(1, 3, 8, 8),),
            m33,
            {
                exir_ops.edge.cortex_m.quantized_conv2d.default: 2,
                exir_ops.edge.cortex_m.quantized_add.default: 1,
            },
            {
                exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default: 2,
                exir_ops.edge.cortex_m.quantized_add.default: 1,
            },
        ),
    )

    for module, inputs, target_config, legacy_ops, explicit_ops in cases:
        legacy = _planned_buffer_sizes(
            module,
            inputs,
            use_explicit_layout=False,
            target_config=target_config,
            expected_ops=legacy_ops,
        )
        explicit = _planned_buffer_sizes(
            module,
            inputs,
            use_explicit_layout=True,
            target_config=target_config,
            expected_ops=explicit_ops,
        )
        assert len(explicit) == len(legacy), type(module).__name__
        assert all(
            explicit_size <= legacy_size
            for explicit_size, legacy_size in zip(explicit, legacy)
        ), type(module).__name__


def test_explicit_nhwc_conv2d_runs_on_fvp():
    x = torch.linspace(-5, 5, steps=1 * 3 * 7 * 10).reshape(1, 3, 7, 10)
    _run_explicit_layout_on_fvp(
        Conv2d(kernel_size=(3, 2), stride=(2, 1), padding=(1, 0)),
        (x,),
        exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default,
    )


def test_explicit_nhwc_depthwise_conv2d_runs_on_fvp():
    x = torch.linspace(-5, 5, steps=1 * 4 * 7 * 10).reshape(1, 4, 7, 10)
    _run_explicit_layout_on_fvp(
        Conv2d(
            4,
            4,
            groups=4,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(1, 0),
        ),
        (x,),
        exir_ops.edge.cortex_m.quantized_depthwise_conv2d_nhwc.default,
    )


def test_explicit_nhwc_transpose_conv2d_runs_on_fvp():
    module = ConvTranspose2d(2, 4, kernel_size=(2, 4), stride=1, padding=0, bias=False)
    module.conv.weight.data.fill_(1.0)
    x = torch.linspace(-8, 8, steps=1 * 2 * 5 * 5).reshape(1, 2, 5, 5)
    _run_explicit_layout_on_fvp(
        module,
        (x,),
        exir_ops.edge.cortex_m.quantized_transpose_conv2d_nhwc.default,
    )


def test_explicit_nhwc_avg_pool2d_runs_on_fvp():
    x = torch.linspace(-5, 5, steps=1 * 3 * 7 * 9).reshape(1, 3, 7, 9)
    _run_explicit_layout_on_fvp(
        ConvPoolConv(
            torch.nn.AvgPool2d(kernel_size=(2, 3), stride=(2, 1), padding=(0, 1))
        ),
        (x,),
        exir_ops.edge.cortex_m.quantized_avg_pool2d_nhwc.default,
    )


def test_explicit_nhwc_max_pool2d_runs_on_fvp():
    x = torch.linspace(-5, 5, steps=1 * 3 * 7 * 9).reshape(1, 3, 7, 9)
    _run_explicit_layout_on_fvp(
        ConvPoolConv(
            torch.nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 1), padding=(0, 1))
        ),
        (x,),
        exir_ops.edge.cortex_m.quantized_max_pool2d_nhwc.default,
    )


def test_explicit_nhwc_pad_runs_on_fvp_with_singleton_height():
    x = torch.linspace(-5, 5, steps=1 * 3 * 1 * 7).reshape(1, 3, 1, 7)
    _run_explicit_layout_on_fvp(
        ConvPadConv(),
        (x,),
        exir_ops.edge.cortex_m.pad_contiguous.default,
    )


def test_explicit_nhwc_channel_broadcast_add_runs_on_fvp():
    x = torch.linspace(-5, 5, steps=1 * 3 * 7 * 9).reshape(1, 3, 7, 9)
    _run_explicit_layout_on_fvp(
        ConvBiasConv(),
        (x,),
        exir_ops.edge.cortex_m.quantized_add.default,
    )


def test_explicit_nhwc_branched_channel_broadcast_runs_on_fvp():
    x = torch.linspace(-5, 5, steps=1 * 3 * 7 * 9).reshape(1, 3, 7, 9)
    _run_explicit_layout_on_fvp(
        ConvBiasBranched(),
        (x,),
        exir_ops.edge.cortex_m.quantized_add.default,
    )


def test_explicit_nhwc_channel_broadcast_mul_runs_on_fvp():
    x = torch.linspace(-5, 5, steps=1 * 3 * 7 * 9).reshape(1, 3, 7, 9)
    _run_explicit_layout_on_fvp(
        ConvMulConv(),
        (x,),
        exir_ops.edge.cortex_m.quantized_mul.default,
    )


def test_aot_explicit_layout_conv1d_runs_on_fvp():
    from types import SimpleNamespace

    from executorch.backends.arm.scripts.aot_arm_compiler import _to_edge_cortex_m
    from executorch.backends.cortex_m.test.tester import CortexMSerialize

    module = Conv1d().eval()
    inputs = (torch.linspace(-5, 5, steps=1 * 2 * 8).reshape(1, 2, 8),)
    exported_program = torch.export.export(module, inputs, strict=True)
    target_config = CortexMTargetConfig(cpu=CortexM.M55)
    model_quant, edge, runtime_inputs = _to_edge_cortex_m(
        exported_program,
        SimpleNamespace(
            cortex_m_explicit_layout=True,
            quantize=True,
            strict_export=True,
        ),
        exported_program.module(),
        inputs,
        None,
        target_config,
    )
    program = edge.exported_program()

    assert model_quant is not None
    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 1
    assert all(
        getattr(node.target, "namespace", None) != "dim_order_ops"
        for node in program.graph.nodes
        if node.op == "call_function"
    )

    serialized = CortexMSerialize(target_config)
    serialized.run(edge.to_executorch())
    [actual] = serialized.run_artifact(runtime_inputs)
    expected = model_quant(*runtime_inputs)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=1e-3)


def _lower_nhwc_io(module, inputs, target_config=None):
    tester = CortexMTester(
        module,
        inputs,
        target_config=target_config,
        use_explicit_layout=True,
        use_nhwc_io=True,
    )
    tester.quantize().export().to_edge().run_passes()
    return tester


def test_nhwc_io_removes_the_boundary_transposes():
    x = torch.randn(1, 3, 8, 8)
    tester = _lower_nhwc_io(Conv2d(), (x,))
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()

    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 1
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 0
    assert program.module()(x.permute(0, 2, 3, 1).contiguous()).shape == torch.Size(
        [1, 8, 8, 4]
    )


def test_nhwc_io_reports_the_contract():
    tester = _lower_nhwc_io(Conv2d(), (torch.randn(1, 3, 8, 8),))
    contract = tester.boundary_layout_contract

    assert contract.inputs == {0: (0, 2, 3, 1)}
    assert contract.outputs == {0: (0, 3, 1, 2)}


def test_nhwc_io_leaves_layout_free_models_alone():
    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 4)

        def forward(self, x):
            return self.linear(x)

    tester = _lower_nhwc_io(Linear(), (torch.randn(2, 8),))

    assert not tester.boundary_layout_contract


def test_nhwc_io_keeps_a_shared_input_on_one_contract_entry():
    x = torch.randn(1, 3, 8, 8)
    tester = _lower_nhwc_io(ConvForkAdd(), (x,))
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()

    assert tester.boundary_layout_contract.inputs == {0: (0, 2, 3, 1)}
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 0


def test_nhwc_io_requires_explicit_layout():
    with pytest.raises(ValueError, match="use_explicit_layout"):
        CortexMPassManager(None, use_nhwc_io=True)


def test_nhwc_io_does_not_increase_planned_memory():
    # Measured against legacy rather than plain explicit layout. Absorption
    # removes two tensors, which reshuffles what the greedy planner packs
    # where; on Conv2d that happens to cost 128 bytes against explicit even
    # though there is strictly less to place. Legacy is the bar that matters.
    torch.manual_seed(0)
    m55 = CortexMTargetConfig(cpu=CortexM.M55)
    for module, inputs in (
        (Conv2d(), (torch.randn(1, 3, 8, 8),)),
        (TwoConv2d(), (torch.randn(1, 3, 8, 8),)),
        (ConvPoolConv(torch.nn.AvgPool2d(2, 2)), (torch.randn(1, 3, 8, 8),)),
        (ConvPoolConv(torch.nn.MaxPool2d(2, 2)), (torch.randn(1, 3, 8, 8),)),
        (ConvForkAdd(), (torch.randn(1, 3, 8, 8),)),
    ):
        legacy = _planned_buffer_sizes(
            module,
            inputs,
            use_explicit_layout=False,
            target_config=m55,
            expected_ops={},
        )
        tester = _lower_nhwc_io(
            copy.deepcopy(module).eval(),
            tuple(value.clone() for value in inputs),
            target_config=m55,
        )
        tester.to_executorch()
        program = tester.get_artifact(StageType.TO_EXECUTORCH).executorch_program
        nhwc_io = tuple(program.execution_plan[0].non_const_buffer_sizes)

        assert len(nhwc_io) == len(legacy), type(module).__name__
        assert all(
            nhwc_size <= legacy_size for nhwc_size, legacy_size in zip(nhwc_io, legacy)
        ), type(module).__name__


def test_nhwc_io_conv2d_runs_on_fvp():
    x = torch.linspace(-5, 5, steps=1 * 3 * 7 * 10).reshape(1, 3, 7, 10)
    tester = _lower_nhwc_io(
        Conv2d(kernel_size=(3, 2), stride=(2, 1), padding=(1, 0)), (x,)
    )
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 0

    tester.to_executorch().serialize()
    tester.run_method_and_compare_outputs(inputs=(x,), qtol=2)


def test_nhwc_io_conv_pool_conv_runs_on_fvp():
    x = torch.linspace(-5, 5, steps=1 * 3 * 8 * 8).reshape(1, 3, 8, 8)
    tester = _lower_nhwc_io(ConvPoolConv(torch.nn.MaxPool2d(2, 2)), (x,))
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 0

    tester.to_executorch().serialize()
    tester.run_method_and_compare_outputs(inputs=(x,), qtol=2)
