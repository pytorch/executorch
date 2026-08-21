# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.passes.scratch_buffer_sizes import (
    required_cmsis_nn_buffer_sizes,
)
from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.backends.cortex_m.test.tester import CortexMTester
from executorch.backends.test.harness.stages import RunPasses, StageType
from executorch.exir.dialects._ops import ops as exir_ops


def _int8_values(shape):
    values = torch.arange(math.prod(shape), dtype=torch.int32)
    return (values.remainder(7) - 3).to(torch.int8).reshape(shape)


def _run_on_fvp(
    module,
    x,
    target,
    target_config: CortexMTargetConfig,
    scratch_count=0,
    atol=1e-3,
):
    sizing_inputs = (x,) + tuple(
        torch.empty(0, dtype=torch.uint8) for _ in range(scratch_count)
    )
    sizing_tester = CortexMTester(module, sizing_inputs, target_config=target_config)
    sizing_tester.export().to_edge()
    sizing_program = sizing_tester.get_artifact(StageType.TO_EDGE).exported_program()
    [node] = [
        node
        for node in sizing_program.graph.nodes
        if node.op == "call_function" and node.target == target
    ]

    if scratch_count:
        scratch_sizes = required_cmsis_nn_buffer_sizes(node, target_config.backend)
        assert scratch_sizes is not None
        assert len(scratch_sizes) == scratch_count
    else:
        scratch_sizes = []

    inputs = (x,) + tuple(
        torch.empty(size, dtype=torch.uint8) for size in scratch_sizes
    )
    tester = CortexMTester(module, inputs, target_config=target_config)
    tester.export().to_edge()
    program = tester.get_artifact(StageType.TO_EDGE).exported_program()
    assert (
        sum(
            node.op == "call_function" and node.target == target
            for node in program.graph.nodes
        )
        == 1
    )
    # The graph already contains the runtime operator; only advance the harness stage.
    tester.run_passes(RunPasses(CortexMPassManager, pass_list=[]))
    tester.to_executorch().serialize()
    tester.run_method_and_compare_outputs(inputs=inputs, atol=atol)


class Conv2dNhwc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", _int8_values((4, 3, 2, 3)))
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


class GroupedConv2dNhwc(torch.nn.Module):
    # OHWI weight with C_in / groups in the last dimension, so the kernel sees
    # an input channel count that differs from the filter's.
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", _int8_values((4, 2, 3, 2)))
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
        self.register_buffer("weight", _int8_values((1, 3, 2, 4)))
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
        self.register_buffer("weight", _int8_values((4, 2, 4, 2)))
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


class PadNhwc(torch.nn.Module):
    def forward(self, x):
        return torch.ops.cortex_m.pad_contiguous.default(
            x,
            [0, 1, 2, 0],
            [0, 2, 1, 0],
            -7,
        )


class AddNhwc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", _int8_values((1, 1, 1, 3)))

    def forward(self, x):
        return torch.ops.cortex_m.quantized_add.default(
            x,
            0,
            1 << 30,
            -1,
            self.bias,
            0,
            1 << 30,
            -1,
            0,
            1 << 30,
            -1,
            -128,
            127,
        )


class MulNhwc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", _int8_values((1, 1, 1, 3)))

    def forward(self, x):
        return torch.ops.cortex_m.quantized_mul.default(
            x,
            0,
            self.bias,
            0,
            0,
            1 << 30,
            -1,
        )


def test_conv2d_nhwc_runs_on_fvp(cortex_m_target):
    _run_on_fvp(
        Conv2dNhwc(),
        _int8_values((1, 7, 10, 3)),
        exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default,
        cortex_m_target,
        scratch_count=1,
    )


def test_grouped_conv2d_nhwc_runs_on_fvp(cortex_m_target):
    _run_on_fvp(
        GroupedConv2dNhwc(),
        _int8_values((1, 7, 10, 4)),
        exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default,
        cortex_m_target,
        scratch_count=1,
    )


def test_depthwise_conv2d_nhwc_runs_on_fvp(cortex_m_target):
    _run_on_fvp(
        DepthwiseConv2dNhwc(),
        _int8_values((1, 7, 10, 4)),
        exir_ops.edge.cortex_m.quantized_depthwise_conv2d_nhwc.default,
        cortex_m_target,
        scratch_count=1,
    )


def test_transpose_conv2d_nhwc_runs_on_fvp(cortex_m_target):
    _run_on_fvp(
        TransposeConv2dNhwc(),
        _int8_values((1, 5, 6, 2)),
        exir_ops.edge.cortex_m.quantized_transpose_conv2d_nhwc.default,
        cortex_m_target,
        scratch_count=2,
    )


def test_avg_pool2d_nhwc_runs_on_fvp(cortex_m_target):
    _run_on_fvp(
        AvgPool2dNhwc(),
        _int8_values((1, 7, 9, 3)),
        exir_ops.edge.cortex_m.quantized_avg_pool2d_nhwc.default,
        cortex_m_target,
        scratch_count=1,
        atol=1,
    )


def test_max_pool2d_nhwc_runs_on_fvp(cortex_m_target):
    _run_on_fvp(
        MaxPool2dNhwc(),
        _int8_values((1, 7, 9, 3)),
        exir_ops.edge.cortex_m.quantized_max_pool2d_nhwc.default,
        cortex_m_target,
    )


def test_pad_contiguous_runs_on_fvp_with_singleton_height(cortex_m_target):
    _run_on_fvp(
        PadNhwc(),
        _int8_values((1, 1, 7, 3)),
        exir_ops.edge.cortex_m.pad_contiguous.default,
        cortex_m_target,
    )


def test_channel_broadcast_add_nhwc_runs_on_fvp(cortex_m_target):
    _run_on_fvp(
        AddNhwc(),
        _int8_values((1, 5, 7, 3)),
        exir_ops.edge.cortex_m.quantized_add.default,
        cortex_m_target,
        atol=1,
    )


def test_channel_broadcast_mul_nhwc_runs_on_fvp(cortex_m_target):
    _run_on_fvp(
        MulNhwc(),
        _int8_values((1, 5, 7, 3)),
        exir_ops.edge.cortex_m.quantized_mul.default,
        cortex_m_target,
        atol=1,
    )
