# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.cortex_m.test.tester import CortexMSerialize
from executorch.exir.dialects._ops import ops as exir_ops


_LEGACY_SPATIAL_OPS = {
    exir_ops.edge.cortex_m.quantized_conv2d.default,
    exir_ops.edge.cortex_m.quantized_depthwise_conv2d.default,
    exir_ops.edge.cortex_m.quantized_transpose_conv2d.default,
    exir_ops.edge.cortex_m.quantized_avg_pool2d.default,
    exir_ops.edge.cortex_m.quantized_max_pool2d.default,
}

# Temporary dual-mode acceptance coverage; see "Explicit-layout migration" in
# the Cortex-M README.


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


def _compile(module, inputs, *, explicit_layout: bool, quantize: bool = True):
    from executorch.backends.arm.scripts.aot_arm_compiler import _to_edge_cortex_m

    exported_program = torch.export.export(module, inputs, strict=True)
    return _to_edge_cortex_m(
        exported_program,
        SimpleNamespace(
            cortex_m_explicit_layout=explicit_layout,
            quantize=quantize,
            strict_export=True,
        ),
        exported_program.module(),
        inputs,
        None,
        CortexMTargetConfig(cpu=CortexM.M55),
    )


def _count(exported_program, target) -> int:
    return sum(node.target == target for node in exported_program.graph.nodes)


@pytest.mark.parametrize(
    "explicit_layout,expected,unexpected",
    [
        (
            False,
            exir_ops.edge.cortex_m.quantized_conv2d.default,
            exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default,
        ),
        (
            True,
            exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default,
            exir_ops.edge.cortex_m.quantized_conv2d.default,
        ),
    ],
)
def test_aot_layout_mode_selects_operator_family(explicit_layout, expected, unexpected):
    _, edge, _ = _compile(
        Conv2d().eval(),
        (torch.randn(1, 3, 8, 8),),
        explicit_layout=explicit_layout,
    )
    program = edge.exported_program()

    assert _count(program, expected) == 1
    assert _count(program, unexpected) == 0


def test_aot_explicit_layout_requires_quantization():
    with pytest.raises(RuntimeError, match="requires --quantize"):
        _compile(
            Conv2d().eval(),
            (torch.randn(1, 3, 8, 8),),
            explicit_layout=True,
            quantize=False,
        )


def _assert_explicit_copy_ceiling(module, inputs, ceiling):
    _, edge, _ = _compile(module, inputs, explicit_layout=True)
    program = edge.exported_program()
    copies = [
        node
        for node in program.graph.nodes
        if node.target
        in {
            exir_ops.edge.cortex_m.transpose.default,
            exir_ops.edge.aten.view_copy.default,
        }
    ]

    assert len(copies) <= ceiling
    assert all(
        node.args[0].meta["val"].dtype == torch.int8
        for node in copies
        if node.target == exir_ops.edge.cortex_m.transpose.default
    )
    assert not any(node.target in _LEGACY_SPATIAL_OPS for node in program.graph.nodes)
    assert not any(
        getattr(node.target, "namespace", None) == "channels_last"
        for node in program.graph.nodes
    )


def test_mobilenet_v2_explicit_copy_ceiling():
    torchvision = pytest.importorskip("torchvision")
    _assert_explicit_copy_ceiling(
        torchvision.models.mobilenet_v2(weights=None).eval(),
        (torch.randn(1, 3, 224, 224),),
        ceiling=2,
    )


def test_resnet8_explicit_copy_ceiling():
    from executorch.examples.models.mlperf_tiny.resnet8 import ResNet8

    _assert_explicit_copy_ceiling(
        ResNet8().eval(),
        (torch.rand(1, 3, 32, 32) * 2 - 1,),
        ceiling=2,
    )


def test_silero_explicit_copy_ceiling():
    from executorch.examples.models.silero_vad.export_silero_vad import (
        CONTEXT_SIZE,
        HIDDEN_DIM,
        SileroVAD16k,
        WINDOW_SIZE,
    )

    _assert_explicit_copy_ceiling(
        SileroVAD16k().eval(),
        (
            torch.randn(1, CONTEXT_SIZE + WINDOW_SIZE),
            torch.zeros(2, 1, HIDDEN_DIM),
        ),
        ceiling=12,
    )


def test_explicit_conv1d_runs_on_fvp():
    inputs = (torch.linspace(-5, 5, steps=16).reshape(1, 2, 8),)
    model_quant, edge, runtime_inputs = _compile(
        Conv1d().eval(), inputs, explicit_layout=True
    )
    program = edge.exported_program()
    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 1

    serialized = CortexMSerialize(CortexMTargetConfig(cpu=CortexM.M55))
    serialized.run(edge.to_executorch())
    [actual] = serialized.run_artifact(runtime_inputs)
    expected = model_quant(*runtime_inputs)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=1e-3)
