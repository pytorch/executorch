# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest
import torch
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
    CortexMPassManager,
    LiftConstantTensorsPass,
)
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.cortex_m.test.tester import CortexMTester
from executorch.backends.test.harness.stages import Quantize, RunPasses, StageType
from executorch.backends.transforms.remove_unused_constants_pass import (
    RemoveUnusedConstantsPass,
)
from executorch.exir import to_edge
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


class NHWCPaddedConv(torch.nn.Module):
    def __init__(self, channels, shared_pad):
        super().__init__()
        out_channels = 64 if channels == 1 else 8
        self.conv = torch.nn.Conv2d(channels, out_channels, (10, 4), stride=2)
        self.shared_pad = shared_pad

    def forward(self, x):
        padded = torch.nn.functional.pad(x.permute(0, 3, 1, 2), (1, 1, 4, 5))
        output = self.conv(padded).permute(0, 2, 3, 1)
        return (output, padded) if self.shared_pad else output


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

    for program in (legacy_program, explicit_program):
        assert all(
            node.users for node in program.graph.nodes if node.op == "placeholder"
        )

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


@pytest.mark.parametrize(
    "passes,lifted,pruned",
    [
        pytest.param([], False, False, id="empty"),
        pytest.param([LiftConstantTensorsPass], True, False, id="lift"),
        pytest.param(
            [LiftConstantTensorsPass, RemoveUnusedConstantsPass],
            True,
            True,
            id="lift_and_prune",
        ),
    ],
)
def test_constant_cleanup_respects_pass_list_and_lift_order(passes, lifted, pruned):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("_lifted_tensor_constant0", torch.tensor([1.0]))
            self.register_buffer("_lifted_tensor_constant1", torch.tensor([2.0]))

        def forward(self, x):
            return x + self._lifted_tensor_constant1

    inputs = (torch.zeros(1),)
    program = to_edge(torch.export.export(Model(), inputs)).exported_program()
    assert "_lifted_tensor_constant0" in program.graph_signature.buffers
    program.graph_module.register_buffer("new_tensor", torch.tensor([3.0]))
    output = next(node for node in program.graph.nodes if node.op == "output")
    original = output.args[0][0]
    with program.graph.inserting_before(original):
        constant = program.graph.get_attr("new_tensor")
        constant.meta = original.meta.copy()
        result = program.graph.call_function(
            exir_ops.edge.aten.add.Tensor, (original.args[1], constant)
        )
        result.meta = original.meta.copy()
    original.replace_input_with(original.args[1], result)
    program.graph_module.recompile()
    program.validate()

    program = CortexMPassManager(program, passes=passes).transform()
    program.validate()
    assert ("_lifted_tensor_constant0" in program.graph_signature.buffers) == (
        not pruned
    )
    assert sum(node.op == "get_attr" for node in program.graph.nodes) == (not lifted)
    torch.testing.assert_close(program.module()(*inputs), torch.tensor([5.0]))
    torch.testing.assert_close(
        program.state_dict["_lifted_tensor_constant1"], torch.tensor([2.0])
    )


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


def _lower_nhwc_padded_conv(channels, shared_pad):
    torch.manual_seed(7)
    tester = _run_explicit_layout_passes(
        CortexMTester(
            NHWCPaddedConv(channels, shared_pad).eval(),
            (torch.randn(1, 49, 10, channels),),
        )
    )
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    assert _count(program, exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default) == 1
    assert _count(program, exir_ops.edge.cortex_m.pad.default) == int(shared_pad)
    if not shared_pad:
        assert _count(program, exir_ops.edge.cortex_m.transpose.default) == 0
    return tester


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("shared_pad", [False, True])
def test_explicit_layout_fuses_same_padding(channels, shared_pad):
    tester = _lower_nhwc_padded_conv(channels, shared_pad)
    tester.run_method_and_compare_outputs(inputs=tester.example_inputs, qtol=1)


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("shared_pad", [False, True])
def test_implementation_explicit_layout_fuses_same_padding(channels, shared_pad):
    tester = _lower_nhwc_padded_conv(channels, shared_pad)
    tester.to_executorch().serialize()
    tester.run_method_and_compare_outputs(inputs=tester.example_inputs, qtol=1)


@pytest.mark.parametrize("hardtanh", [False, True])
def test_implementation_transpose_conv2d_strided_pointwise(hardtanh):
    torch.manual_seed(0)
    inputs = (torch.randn(2, 4, 9).unsqueeze(2),)
    model = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(4, 4, (1, 1), stride=(1, 2)),
        torch.nn.Hardtanh(-0.5, 0.5) if hardtanh else torch.nn.Identity(),
    ).eval()
    tester = _run_explicit_layout_passes(CortexMTester(model, inputs))
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    assert (
        _count(program, exir_ops.edge.cortex_m.quantized_transpose_conv2d_nhwc.default)
        == 1
    )
    tester.to_executorch().serialize()
    tester.run_method_and_compare_outputs(inputs=inputs, qtol=1)


def test_explicit_layout_rejects_unsupported_spatial_operator():
    tester = CortexMTester(UnsupportedAvgPool(), (torch.randn(1, 3, 8, 8),))

    with pytest.raises(Exception) as caught:
        _run_explicit_layout_passes(tester)

    assert caught.value.__cause__ is not None
    assert "NHWC-eligible" in str(caught.value.__cause__)
