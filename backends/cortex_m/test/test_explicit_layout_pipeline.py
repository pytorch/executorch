# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest
import torch
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.cortex_m.test.tester import CortexMTester
from executorch.backends.test.harness.stages import Quantize, RunPasses, StageType
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import Node

# Temporary opt-in coverage. Move these invariants into the standard Cortex-M
# tests when this pass manager becomes the default.


class Conv2d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class Conv1d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 4, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class ConvPadConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 5, 3, padding=1)

    def forward(self, x):
        return self.conv2(torch.nn.functional.pad(self.conv1(x), (1, 1, 1, 1)))


class UnsupportedAvgPool(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.avg_pool2d(
            x, kernel_size=2, stride=2, divisor_override=3
        )


def _count(exported_program, target) -> int:
    return sum(node.target == target for node in exported_program.graph.nodes)


def _run_explicit_layout_pass_manager(tester: CortexMTester) -> CortexMTester:
    target_config = CortexMTargetConfig(cpu=CortexM.M55)
    tester.run_passes(
        RunPasses(
            partial(
                CortexMPassManager,
                target_config=target_config,
                use_explicit_layout=True,
            ),  # type: ignore[arg-type]
            CortexMPassManager.explicit_layout_pass_list,  # type: ignore[arg-type]
        )
    )
    return tester


def _run_explicit_layout_passes(tester: CortexMTester) -> CortexMTester:
    tester.quantize(Quantize(CortexMQuantizer(use_explicit_layout=True)))
    tester.export().to_edge()
    return _run_explicit_layout_pass_manager(tester)


def test_layout_pipelines_select_distinct_spatial_operators():
    legacy_input = torch.randn(1, 3, 8, 8).to(memory_format=torch.channels_last)
    legacy = CortexMTester(
        Conv2d().eval().to(memory_format=torch.channels_last),
        (legacy_input,),
    )
    legacy.quantize().export().to_edge().run_passes()
    legacy_program = legacy.get_artifact(StageType.RUN_PASSES).exported_program()

    explicit = _run_explicit_layout_passes(
        CortexMTester(Conv2d().eval(), (torch.randn(1, 3, 8, 8),))
    )
    explicit_program = explicit.get_artifact(StageType.RUN_PASSES).exported_program()

    assert _count(legacy_program, exir_ops.edge.cortex_m.quantized_conv2d.default) == 1
    assert (
        _count(
            legacy_program,
            exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default,
        )
        == 0
    )
    assert (
        _count(explicit_program, exir_ops.edge.cortex_m.quantized_conv2d.default) == 0
    )
    assert (
        _count(
            explicit_program,
            exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default,
        )
        == 1
    )
    assert _count(explicit_program, exir_ops.edge.cortex_m.transpose.default) == 2


def test_conv1d_is_quantized_before_layout_conversion():
    tester = CortexMTester(Conv1d().eval(), (torch.randn(1, 2, 8),))
    tester.quantize(Quantize(CortexMQuantizer(use_explicit_layout=True)))
    quantized = tester.get_artifact(StageType.QUANTIZE)
    [conv1d] = [
        node
        for node in quantized.graph.nodes
        if node.target == torch.ops.aten.conv1d.default
    ]

    weight = conv1d.args[1]
    assert isinstance(weight, Node)
    assert (
        weight.target == torch.ops.quantized_decomposed.dequantize_per_channel.default
    )

    tester.export().to_edge()
    _run_explicit_layout_pass_manager(tester)
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()

    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 1
    assert _count(program, exir_ops.edge.aten.convolution.default) == 0


def test_explicit_layout_reuses_pad():
    tester = _run_explicit_layout_passes(
        CortexMTester(ConvPadConv().eval(), (torch.randn(1, 3, 8, 8),))
    )
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()

    assert _count(program, exir_ops.edge.cortex_m.pad.default) == 1


def test_explicit_layout_rejects_unsupported_spatial_operator():
    tester = CortexMTester(UnsupportedAvgPool(), (torch.randn(1, 3, 8, 8),))

    with pytest.raises(Exception) as caught:
        _run_explicit_layout_passes(tester)

    assert caught.value.__cause__ is not None
    assert "NHWC-eligible" in str(caught.value.__cause__)
