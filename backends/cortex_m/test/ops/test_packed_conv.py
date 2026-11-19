# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import executorch.backends.cortex_m.ops.operators  # noqa: F401

import pytest
import torch
from executorch.backends.cortex_m.library import cmsis_nn
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.cortex_m.test.ops.nhwc_test_utils import (
    int8_values,
    run_on_fvp,
)
from executorch.backends.cortex_m.test.tester import (
    CortexMQuantize,
    CortexMRunPasses,
    CortexMTester,
)
from executorch.backends.test.harness.stages import StageType
from executorch.backends.transforms.utils import get_param_tensor
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.mark.parametrize("nhwc", [False, True])
def test_original_conv_schema(nhwc):
    op = (
        torch.ops.cortex_m.quantized_conv2d_nhwc
        if nhwc
        else torch.ops.cortex_m.quantized_conv2d
    )
    assert len(op.default._schema.arguments) == 13
    assert len(op.out._schema.arguments) == 14
    assert op.out._schema.arguments[-1].name == "out"


@pytest.mark.parametrize(
    "shape,groups", [((7, 1, 1, 19), 1), ((21, 3, 2, 19), 3), ((8, 1, 3, 8), 2)]
)
@pytest.mark.parametrize("bias", [False, True])
def test_prepare_weights(shape, groups, bias):
    weight = int8_values(shape)
    original_bias = torch.arange(shape[0], dtype=torch.int32) if bias else None
    packed, effective = cmsis_nn.convolve_s8_prepare_weights(
        weight, -7, original_bias, groups
    )
    rows = shape[0] // groups
    reduction = math.prod(shape[1:])
    restored = packed.reshape(groups, (rows + 3) // 4, (reduction + 15) // 16, 4, 16)
    restored = restored.permute(0, 1, 3, 2, 4).reshape(
        groups, ((rows + 3) // 4) * 4, -1
    )
    torch.testing.assert_close(
        restored[:, :rows, :reduction], weight.reshape(groups, rows, reduction)
    )
    assert torch.count_nonzero(restored[:, rows:]) == 0
    assert torch.count_nonzero(restored[:, :, reduction:]) == 0
    expected = weight.int().reshape(shape[0], -1).sum(1).int() * -7
    if bias:
        expected += original_bias
    torch.testing.assert_close(effective, expected)
    assert packed.dtype == torch.int8 and packed.dim() == 1
    assert effective.dtype == torch.int32 and effective.shape == (shape[0],)


class PackedConv(torch.nn.Module):
    def __init__(
        self, shape, groups, bias, nhwc, stride=(1, 1), padding=(0, 0), dilation=(1, 1)
    ):
        super().__init__()
        self.shape = shape
        self.groups = groups
        self.nhwc = nhwc
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        weight = int8_values(shape)
        original_bias = torch.arange(shape[0], dtype=torch.int32) if bias else None
        packed, effective = cmsis_nn.convolve_s8_prepare_weights(
            weight, 3, original_bias, groups
        )
        self.register_buffer("weight", packed)
        self.register_buffer("effective_bias", effective)
        self.register_buffer(
            "multipliers", torch.full((shape[0],), 1 << 30, dtype=torch.int32)
        )
        self.register_buffer("shifts", torch.full((shape[0],), -2, dtype=torch.int32))

    def forward(self, x, scratch):
        op = (
            torch.ops.cortex_m.quantized_conv2d_packed_nhwc.default
            if self.nhwc
            else torch.ops.cortex_m.quantized_conv2d_packed.default
        )
        return op(
            x,
            self.weight,
            self.effective_bias,
            self.stride,
            self.padding,
            self.dilation,
            3,
            0,
            self.multipliers,
            self.shifts,
            -128,
            127,
            scratch,
            filter_shape=self.shape,
        )


@pytest.mark.parametrize("nhwc", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize(
    "shape,groups,stride,padding,dilation",
    [
        ((7, 1, 1, 19), 1, (1, 1), (0, 0), (1, 1)),
        ((21, 3, 2, 3), 3, (2, 1), (1, 0), (1, 2)),
    ],
)
def test_packed_reference(nhwc, bias, batch, shape, groups, stride, padding, dilation):
    module = PackedConv(shape, groups, bias, nhwc, stride, padding, dilation)
    x = int8_values((batch, shape[3] * groups, 7, 9))
    original_weight = int8_values(shape).double().permute(0, 3, 1, 2)
    original_bias = torch.arange(shape[0], dtype=torch.int32) if bias else None
    acc = torch.nn.functional.conv2d(
        x.double() + 3,
        original_weight,
        original_bias.double() if original_bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    from executorch.backends.cortex_m.ops.operators import requantize_cmsis

    expected = requantize_cmsis(acc.int(), 1 << 30, -2).clamp(-128, 127).to(torch.int8)
    if nhwc:
        x = x.permute(0, 2, 3, 1).contiguous()
        expected = expected.permute(0, 2, 3, 1).contiguous()
    else:
        x = x.to(memory_format=torch.channels_last)
    torch.testing.assert_close(module(x, torch.empty(0, dtype=torch.uint8)), expected)
    with FakeTensorMode(allow_non_fake_inputs=True):
        assert module(x, torch.empty(0, dtype=torch.uint8)).shape == expected.shape


@pytest.mark.parametrize("nhwc", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_unpacked_original_api(nhwc, bias):
    shape = (21, 3, 2, 3)
    model = PackedConv(shape, 3, bias, nhwc)
    x = int8_values((2, 9, 7, 9))
    x = (
        x.permute(0, 2, 3, 1).contiguous()
        if nhwc
        else x.to(memory_format=torch.channels_last)
    )
    scratch = torch.empty(0, dtype=torch.uint8)
    expected = model(x, scratch)
    op = (
        torch.ops.cortex_m.quantized_conv2d_nhwc.default
        if nhwc
        else torch.ops.cortex_m.quantized_conv2d.default
    )
    args = (
        x,
        int8_values(shape),
        torch.arange(shape[0], dtype=torch.int32) if bias else None,
        model.stride,
        model.padding,
        model.dilation,
        3,
        0,
        model.multipliers,
        model.shifts,
        -128,
        127,
        scratch,
    )
    torch.testing.assert_close(op(*args), expected)
    with FakeTensorMode(allow_non_fake_inputs=True):
        assert op(*args).shape == expected.shape


@pytest.mark.parametrize("nhwc", [False, True])
@pytest.mark.parametrize(
    "shape,groups,stride,padding",
    [
        ((7, 1, 1, 19), 1, (1, 1), (0, 0)),
        ((21, 3, 2, 3), 3, (2, 1), (1, 0)),
        ((7, 1, 3, 3), 1, (1, 2), (0, 1)),
        ((7, 1, 3, 8), 1, (1, 2), (0, 1)),
    ],
)
def test_implementation_packed(nhwc, shape, groups, stride, padding, cortex_m_target):
    if cortex_m_target.backend != cmsis_nn.Backend.MVE:
        pytest.skip("Packed effective-bias convolution requires MVE")
    model = PackedConv(shape, groups, True, nhwc, stride, padding)
    height = 1 if shape[1] == 1 and shape[2] > 1 else 7
    x = int8_values((2, shape[3] * groups, height, 9))
    x = (
        x.permute(0, 2, 3, 1).contiguous()
        if nhwc
        else x.to(memory_format=torch.channels_last)
    )
    target = (
        exir_ops.edge.cortex_m.quantized_conv2d_packed_nhwc.default
        if nhwc
        else exir_ops.edge.cortex_m.quantized_conv2d_packed.default
    )
    run_on_fvp(model, x, target, cortex_m_target, 1)


@pytest.mark.parametrize(
    "backend", [cmsis_nn.Backend.MVE, cmsis_nn.Backend.DSP, cmsis_nn.Backend.SCALAR]
)
@pytest.mark.parametrize("groups", [1, 3])
@pytest.mark.parametrize("nhwc", [False, True])
def test_packed_lowering(backend, groups, nhwc):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(9, 21, (3, 2), groups=groups, padding=(1, 0), stride=(2, 1))
    ).eval()
    x = torch.randn(2, 9, 7, 9)
    inputs = (x if nhwc else x.to(memory_format=torch.channels_last),)
    config = CortexMTargetConfig(cpu=CortexM.M55, isa=backend)
    tester = CortexMTester(model, inputs, target_config=config)
    tester.quantize(
        CortexMQuantize(use_explicit_layout=nhwc)
    ).export().to_edge().run_passes(
        CortexMRunPasses(target_config=config, use_explicit_layout=nhwc)
    )
    tester.run_method_and_compare_outputs(inputs=inputs, qtol=1)
    ep = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    [conv] = [n for n in ep.graph.nodes if "cortex_m.quantized_conv2d" in str(n.target)]
    weight = get_param_tensor(ep, conv.args[1])
    if backend == cmsis_nn.Backend.MVE:
        expected = (
            exir_ops.edge.cortex_m.quantized_conv2d_packed_nhwc.default
            if nhwc
            else exir_ops.edge.cortex_m.quantized_conv2d_packed.default
        )
        assert conv.target == expected
        assert list(conv.kwargs["filter_shape"]) == [21, 3, 2, 9 // groups]
        assert weight.dim() == 1
        assert get_param_tensor(ep, conv.args[2]).shape == (21,)
        constants = (*ep.state_dict.values(), *ep.constants.values())
        assert (
            sum(
                isinstance(value, torch.Tensor) and value.dtype == torch.int8
                for value in constants
            )
            == 1
        )
    else:
        assert weight.shape == (21, 3, 2, 9 // groups)
        expected = (
            exir_ops.edge.cortex_m.quantized_conv2d_nhwc.default
            if nhwc
            else exir_ops.edge.cortex_m.quantized_conv2d.default
        )
        assert conv.target == expected
        assert not conv.kwargs


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("nhwc", [False, True])
@pytest.mark.parametrize(
    "invalid",
    [
        "weights",
        "bias",
        "missing_bias",
        "bias_dtype",
        "shape",
        "empty_shape",
        "groups",
        "unpacked_weights",
    ],
)
def test_invalid_packed_inputs(fake, nhwc, invalid):
    model = PackedConv((7, 1, 1, 19), 1, True, nhwc)
    if invalid == "weights":
        model.weight = model.weight[:-1]
    elif invalid == "bias":
        model.effective_bias = model.effective_bias[:-1]
    elif invalid == "missing_bias":
        model.effective_bias = None
    elif invalid == "bias_dtype":
        model.effective_bias = model.effective_bias.to(torch.int64)
    elif invalid == "shape":
        model.shape = (7, 1, 19)
    elif invalid == "empty_shape":
        model.shape = ()
    elif invalid == "groups":
        model.shape = (7, 1, 1, 3)
    else:
        model.weight = int8_values(model.shape)
    x = int8_values((1, 19, 3, 5))
    if nhwc:
        x = x.permute(0, 2, 3, 1).contiguous()
    else:
        x = x.to(memory_format=torch.channels_last)
    if fake:
        with FakeTensorMode(allow_non_fake_inputs=True), pytest.raises(ValueError):
            model(x, torch.empty(0, dtype=torch.uint8))
    else:
        with pytest.raises(ValueError):
            model(x, torch.empty(0, dtype=torch.uint8))


@pytest.mark.parametrize("nhwc", [False, True])
def test_implementation_packed_lowering(nhwc, cortex_m_target):
    if cortex_m_target.backend != cmsis_nn.Backend.MVE:
        pytest.skip("Packed effective-bias convolution requires MVE")
    model = torch.nn.Sequential(
        torch.nn.Conv2d(9, 21, (3, 2), stride=(2, 1), padding=(1, 0), groups=3)
    ).eval()
    x = torch.randn(2, 9, 7, 9)
    inputs = (x if nhwc else x.to(memory_format=torch.channels_last),)
    tester = CortexMTester(model, inputs, target_config=cortex_m_target)
    tester.quantize(
        CortexMQuantize(use_explicit_layout=nhwc)
    ).export().to_edge().run_passes(
        CortexMRunPasses(target_config=cortex_m_target, use_explicit_layout=nhwc)
    )
    tester.to_executorch().serialize().run_method_and_compare_outputs(
        inputs=inputs, qtol=1
    )


class SharedParametersConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 3)

    def forward(self, x):
        return self.conv(x), self.conv.weight, self.conv.bias


def test_shared_parameters_survive_packing():
    model = SharedParametersConv().eval()
    inputs = (torch.randn(1, 3, 7, 9).to(memory_format=torch.channels_last),)
    tester = CortexMTester(
        model, inputs, target_config=CortexMTargetConfig(cpu=CortexM.M55)
    )
    tester.quantize().export().to_edge()
    reference = (
        tester.get_artifact(StageType.TO_EDGE).exported_program().module()(*inputs)
    )
    tester.run_passes()
    tester.run_method_and_compare_outputs(inputs=inputs, qtol=1)
    ep = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    actual = ep.module()(*inputs)
    torch.testing.assert_close(actual[1], reference[1])
    torch.testing.assert_close(actual[2], reference[2])
    [conv] = [n for n in ep.graph.nodes if "cortex_m.quantized_conv2d" in str(n.target)]
    assert get_param_tensor(ep, conv.args[1]).dim() == 1


@pytest.mark.parametrize("groups", [1, 3])
def test_aot_disabled_lowering(monkeypatch, groups):
    monkeypatch.setattr(cmsis_nn._require_cmsis_nn(), "aot_enabled", False)
    model = torch.nn.Sequential(torch.nn.Conv2d(9, 21, 3, groups=groups)).eval()
    inputs = (torch.randn(2, 9, 7, 9).to(memory_format=torch.channels_last),)
    tester = CortexMTester(
        model, inputs, target_config=CortexMTargetConfig(cpu=CortexM.M55)
    )
    tester.quantize().export().to_edge().run_passes()
    tester.run_method_and_compare_outputs(inputs=inputs, qtol=1)
    ep = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    [conv] = [n for n in ep.graph.nodes if "cortex_m.quantized_conv2d" in str(n.target)]
    assert conv.target == exir_ops.edge.cortex_m.quantized_conv2d.default
    assert not conv.kwargs and conv.args[2] is not None
    assert get_param_tensor(ep, conv.args[1]).shape == (21, 3, 3, 9 // groups)


@pytest.mark.parametrize("nhwc", [False, True])
def test_implementation_aot_disabled_lowering(monkeypatch, nhwc, cortex_m_target):
    monkeypatch.setattr(cmsis_nn._require_cmsis_nn(), "aot_enabled", False)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 7, 3)).eval()
    x = torch.randn(2, 3, 7, 9)
    inputs = (x if nhwc else x.to(memory_format=torch.channels_last),)
    CortexMTester(model, inputs, target_config=cortex_m_target).test_implementation(
        qtol=1, use_explicit_layout=nhwc
    )
