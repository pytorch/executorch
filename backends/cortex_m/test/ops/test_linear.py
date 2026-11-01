# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from contextlib import nullcontext
from dataclasses import dataclass

import pytest
import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a8w4_quantization_config,
)
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    A8W4_PER_CHANNEL_CONFIG,
    CortexMQuantizationConfig,
    INT8_ACTIVATION_PER_TENSOR_QSPEC,
    INT8_PER_CHANNEL_CONFIG,
    INT8_WEIGHT_PER_CHANNEL_TRANSPOSE_QSPEC,
)
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)
from executorch.backends.cortex_m.utils import pack_int4, unpack_int4
from executorch.backends.test.harness.stages import Quantize, StageType
from executorch.backends.transforms.utils import get_param_tensor
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


class CortexMLinear(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(*args, bias=False)
        self.linear.weight.data.fill_(1.0)

    def forward(self, x):
        return self.linear(x)


class CortexMLinearX3(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 4,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 7,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 3,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(*args, bias=False)
        self.linear.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.linear(x)
        x = self.linear(x)
        x = self.linear(x)
        return x


class CortexMLinearBias(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 4,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(*args, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.linear(x)


test_cases = {
    "linear_rank1": McuTestCase(
        model=CortexMLinear(1, 2),
        example_inputs=(torch.Tensor([1]),),
    ),
    "linear_rank2_pos": McuTestCase(
        model=CortexMLinear(1, 2),
        example_inputs=(ramp_tensor(-1, 1, (1, 1)),),
    ),
    "linear_rank3_neg": McuTestCase(
        model=CortexMLinear(5, 3),
        example_inputs=(ramp_tensor(-40, 0, (4, 2, 5)),),
    ),
    "linear_rank4": McuTestCase(
        model=CortexMLinear(16, 32),
        example_inputs=(ramp_tensor(-100, 100, (2, 1, 2, 16)),),
    ),
    "linear_rank5": McuTestCase(
        model=CortexMLinear(4, 3),
        example_inputs=(ramp_tensor(-2, 2, (5, 2, 1, 2, 4)),),
    ),
    "linear_bias": McuTestCase(
        model=CortexMLinearBias(61, 37),
        example_inputs=(ramp_tensor(0, 10, (8, 61)),),
    ),
    "linear_x3": McuTestCase(
        model=CortexMLinearX3(4, 4),
        example_inputs=(ramp_tensor(0, 10, (2, 4)),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_linear(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


@parametrize("test_case", test_cases)
def test_implementation_linear(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(qtol=1)


# ---------------------------------------------------------------------------
# Regression: cortex_m::quantized_linear must pick the right CMSIS-NN input
# convention based on the target ISA. `arm_fully_connected_s8` reads
# kernel_sum (ctx.buf) on MVE/Helium and reads the bias argument on DSP/scalar
# paths; the two are mutually exclusive. Previously the pass unconditionally
# emitted the MVE shape, which silently dropped the bias and input-offset
# terms on every non-MVE build. The regression only showed up when those
# terms dominated the int32 accumulator -- i.e., on small-magnitude inputs.
#
# Coverage strategy: a single ISA-parametrized dialect test verifies the
# numeric output against the float reference (catches the dropped-bias bug
# directly), checks ops_after_transforms to confirm the linear lowered, and
# asserts the post-pass node has the value in the slot the configured ISA
# expects -- the structural guard against a regression that emits zero-valued
# kernel_sum on a no-bias DSP path (numerically inert, but wrong shape).
# An additional implementation test drives each configured target through
# the simulator.
# ---------------------------------------------------------------------------


class _SmallMagnitudeLinear(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 4,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, bias: bool = True):
        super().__init__()
        self.fc = torch.nn.Linear(512, 10, bias=bias)

    def forward(self, x):
        return self.fc(x)


class _SmallMagnitudeLinearNoBias(_SmallMagnitudeLinear):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 3,
    }

    def __init__(self):
        super().__init__(bias=False)


def _small_magnitude_input():
    return torch.rand(1, 512) * 0.002


_small_magnitude_calibration = [(_small_magnitude_input(),) for _ in range(8)]


@dataclass(frozen=True)
class _SmallMagnitudeVariant:
    case: McuTestCase
    target_config: CortexMTargetConfig
    uses_kernel_sum: bool
    has_bias: bool


def _small_magnitude_variant(
    model_cls, cpu: CortexM, *, uses_kernel_sum: bool, has_bias: bool
) -> _SmallMagnitudeVariant:
    return _SmallMagnitudeVariant(
        case=McuTestCase(
            model=model_cls().eval(),
            example_inputs=lambda: (_small_magnitude_input(),),
        ),
        target_config=CortexMTargetConfig(cpu=cpu),
        uses_kernel_sum=uses_kernel_sum,
        has_bias=has_bias,
    )


# bias=True covers the regression directly (the bug dropped the bias term);
# bias=False covers the symmetric case where only the input-offset term is
# missing on the non-MVE paths.
small_magnitude_variants = {
    "mve_bias": _small_magnitude_variant(
        _SmallMagnitudeLinear, CortexM.M55, uses_kernel_sum=True, has_bias=True
    ),
    "dsp_bias": _small_magnitude_variant(
        _SmallMagnitudeLinear, CortexM.M4, uses_kernel_sum=False, has_bias=True
    ),
    "scalar_bias": _small_magnitude_variant(
        _SmallMagnitudeLinear, CortexM.M0PLUS, uses_kernel_sum=False, has_bias=True
    ),
    "mve_nobias": _small_magnitude_variant(
        _SmallMagnitudeLinearNoBias, CortexM.M55, uses_kernel_sum=True, has_bias=False
    ),
    "dsp_nobias": _small_magnitude_variant(
        _SmallMagnitudeLinearNoBias, CortexM.M4, uses_kernel_sum=False, has_bias=False
    ),
    "scalar_nobias": _small_magnitude_variant(
        _SmallMagnitudeLinearNoBias,
        CortexM.M0PLUS,
        uses_kernel_sum=False,
        has_bias=False,
    ),
}


@parametrize("variant", small_magnitude_variants)
def test_dialect_linear_small_magnitude(variant: _SmallMagnitudeVariant):
    tester = CortexMTester(
        variant.case.model,
        variant.case.get_example_inputs(),
        target_config=variant.target_config,
    )
    tester.test_dialect(
        ops_before_transforms=variant.case.model.ops_before_transforms,
        ops_after_transforms=variant.case.model.ops_after_transforms,
        qtol=1,
        calibration_samples=_small_magnitude_calibration,
    )

    # Structural guard: numeric divergence catches the original dropped-bias
    # bug, but a future regression that emits zero-valued kernel_sum on a
    # no-bias DSP/scalar path would be numerically inert. Assert the slot the
    # configured ISA actually consumes is populated and the unused one is None.
    module = tester.get_artifact(StageType.RUN_PASSES).exported_program().module()
    linear_target = exir_ops.edge.cortex_m.quantized_linear.default
    [linear_node] = [
        n
        for n in module.graph.nodes
        if n.op == "call_function" and n.target == linear_target
    ]
    bias_arg, kernel_sum_arg = linear_node.args[2], linear_node.args[3]
    if variant.uses_kernel_sum:
        assert kernel_sum_arg is not None
        assert bias_arg is None
    else:
        assert kernel_sum_arg is None
        if variant.has_bias:
            assert bias_arg is not None
        else:
            assert bias_arg is None


def test_implementation_linear_small_magnitude(cortex_m_target):
    case = McuTestCase(
        model=_SmallMagnitudeLinear().eval(),
        example_inputs=lambda: (_small_magnitude_input(),),
    )
    tester = CortexMTester(
        case.model, case.get_example_inputs(), target_config=cortex_m_target
    )
    tester.test_implementation(qtol=1, calibration_samples=_small_magnitude_calibration)


class CortexMLinearPerChannel(torch.nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        # Spread the row magnitudes so a per-tensor scale would not fit them.
        self.linear.weight.data *= torch.linspace(0.1, 2.0, out_features)[:, None]

    def forward(self, x):
        return self.linear(x)


per_channel_linear_test_cases = {
    "linear_bias": McuTestCase(
        model=CortexMLinearPerChannel(48, 150, bias=True),
        example_inputs=(ramp_tensor(-3, 3, (3, 48)),),
    ),
    "linear_no_bias": McuTestCase(
        model=CortexMLinearPerChannel(32, 20, bias=False),
        example_inputs=(ramp_tensor(-2, 2, (2, 32)),),
    ),
}


def _lower_linear_per_channel(test_case, target_config, config=INT8_PER_CHANNEL_CONFIG):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=target_config
    )
    quantizer = CortexMQuantizer().set_node_target(
        torch.ops.aten.linear.default, config
    )
    tester.quantize(Quantize(quantizer))
    tester.export().to_edge().run_passes()
    return tester


def _linear_nodes(tester):
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    return [n for n in program.graph.nodes if "quantized_linear" in str(n.target)]


@parametrize("test_case", per_channel_linear_test_cases)
@pytest.mark.parametrize("cpu", [CortexM.M55, CortexM.M4, CortexM.M0PLUS])
def test_dialect_linear_per_channel(test_case, cpu):
    tester = _lower_linear_per_channel(test_case, CortexMTargetConfig(cpu=cpu))
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs, qtol=1)

    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    [linear] = _linear_nodes(tester)
    assert linear.target == exir_ops.edge.cortex_m.quantized_linear_per_channel.default
    out_features = test_case.model.linear.out_features
    for requantize_arg in linear.args[7:9]:
        assert get_param_tensor(program, requantize_arg).shape == (out_features,)
    # Per-channel scales make every weight row reach the end of the int8 range.
    weights = get_param_tensor(program, linear.args[1]).to(torch.int32)
    assert set(weights.abs().amax(dim=1).tolist()) <= {127, 128}


@parametrize("test_case", per_channel_linear_test_cases)
def test_implementation_linear_per_channel(test_case, cortex_m_target):
    tester = _lower_linear_per_channel(test_case, cortex_m_target)
    tester.to_executorch().serialize()
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs, qtol=1)


def test_dialect_linear_defaults_to_per_tensor(cortex_m_target):
    case = per_channel_linear_test_cases["linear_bias"]
    tester = CortexMTester(
        case.model, case.example_inputs, target_config=cortex_m_target
    )
    tester.quantize().export().to_edge().run_passes()
    [linear] = _linear_nodes(tester)
    assert linear.target == exir_ops.edge.cortex_m.quantized_linear.default
    assert len(linear.args[7]) == 1


def test_dialect_linear_per_channel_single_output_feature(cortex_m_target):
    # One output feature has one scale, which the per-tensor op serves.
    case = McuTestCase(
        model=CortexMLinearPerChannel(16, 1, bias=True),
        example_inputs=(ramp_tensor(-1, 1, (2, 16)),),
    )
    tester = _lower_linear_per_channel(case, cortex_m_target)
    tester.run_method_and_compare_outputs(inputs=case.example_inputs, qtol=1)
    [linear] = _linear_nodes(tester)
    assert linear.target == exir_ops.edge.cortex_m.quantized_linear.default


def test_dialect_linear_rejects_per_channel_weights_off_axis_0(cortex_m_target):
    # Per-input-feature scales cannot requantize per output feature, so the
    # linear falls through to the default per-tensor config.
    axis_1_config = CortexMQuantizationConfig(
        INT8_ACTIVATION_PER_TENSOR_QSPEC,
        INT8_ACTIVATION_PER_TENSOR_QSPEC,
        INT8_WEIGHT_PER_CHANNEL_TRANSPOSE_QSPEC,
        None,
    )
    case = per_channel_linear_test_cases["linear_no_bias"]
    tester = _lower_linear_per_channel(case, cortex_m_target, axis_1_config)
    tester.run_method_and_compare_outputs(inputs=case.example_inputs, qtol=1)
    [linear] = _linear_nodes(tester)
    assert linear.target == exir_ops.edge.cortex_m.quantized_linear.default


def test_dialect_linear_per_channel_keeps_fused_relu(cortex_m_target):
    # set_node_target selects the linear; the relu after it must still match
    # as part of the linear + relu pattern and fold into the linear.
    case = McuTestCase(
        model=torch.nn.Sequential(
            CortexMLinearPerChannel(24, 12, bias=True), torch.nn.ReLU()
        ),
        example_inputs=(ramp_tensor(-2, 2, (2, 24)),),
    )
    tester = _lower_linear_per_channel(case, cortex_m_target)
    tester.run_method_and_compare_outputs(inputs=case.example_inputs, qtol=1)
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    targets = [str(n.target) for n in program.graph.nodes if n.op == "call_function"]
    assert any("quantized_linear_per_channel" in t for t in targets)
    assert not any("relu" in t for t in targets)


a8w4_linear_test_cases = {
    "linear_bias": per_channel_linear_test_cases["linear_bias"],
    "linear_no_bias": per_channel_linear_test_cases["linear_no_bias"],
    # 27 * 13 = 351 weights: odd, so the last high nibble is padding.
    "linear_odd_weight_count": McuTestCase(
        model=CortexMLinearPerChannel(27, 13, bias=True),
        example_inputs=(ramp_tensor(-2, 2, (2, 27)),),
    ),
    "linear_3d_input": McuTestCase(
        model=CortexMLinearPerChannel(24, 10, bias=True),
        example_inputs=(ramp_tensor(-2, 2, (2, 3, 24)),),
    ),
    "linear_relu": McuTestCase(
        model=torch.nn.Sequential(
            CortexMLinearPerChannel(24, 12, bias=True), torch.nn.ReLU()
        ),
        example_inputs=(ramp_tensor(-2, 2, (2, 24)),),
    ),
}


def _s4_linear_weights(tester):
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    [linear] = _linear_nodes(tester)
    assert linear.target == exir_ops.edge.cortex_m.quantized_linear_s4.default
    multipliers = get_param_tensor(program, linear.args[5])
    in_features = linear.args[0].meta["val"].shape[-1]
    weights = get_param_tensor(program, linear.args[1])
    assert weights.shape == ((multipliers.numel() * in_features + 1) // 2,)
    return multipliers, unpack_int4(weights, (multipliers.numel(), in_features))


@parametrize("test_case", a8w4_linear_test_cases)
@pytest.mark.parametrize("cpu", [CortexM.M55, CortexM.M4, CortexM.M0PLUS])
def test_dialect_linear_a8w4(test_case, cpu):
    tester = _lower_linear_per_channel(
        test_case, CortexMTargetConfig(cpu=cpu), A8W4_PER_CHANNEL_CONFIG
    )
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs, qtol=1)

    _, weights = _s4_linear_weights(tester)
    # Per-channel scales make every weight row reach the end of the int4 grid.
    assert set(weights.to(torch.int32).abs().amax(dim=1).tolist()) == {7}
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    assert not any("relu" in str(n.target) for n in program.graph.nodes)


def _per_tensor_a8w4_config():
    arm_config = get_symmetric_a8w4_quantization_config(
        is_per_channel=False, is_qat=False
    )
    return CortexMQuantizationConfig(
        arm_config.input_activation,
        arm_config.output_activation,
        arm_config.weight,
        arm_config.bias,
    )


def test_dialect_linear_a8w4_per_tensor(cortex_m_target):
    case = per_channel_linear_test_cases["linear_bias"]
    tester = _lower_linear_per_channel(case, cortex_m_target, _per_tensor_a8w4_config())
    tester.run_method_and_compare_outputs(inputs=case.example_inputs, qtol=1)

    multipliers, weights = _s4_linear_weights(tester)
    # One weight scale, repeated for the per-channel kernel.
    assert multipliers.unique().numel() == 1
    assert weights.abs().max() == 7


@pytest.mark.parametrize(
    "cpu,per_channel,expected",
    [
        pytest.param(
            CortexM.M55,
            True,
            exir_ops.edge.cortex_m.quantized_linear_per_channel.default,
            id="m55",
        ),
        pytest.param(
            CortexM.M55,
            False,
            exir_ops.edge.cortex_m.quantized_linear.default,
            id="m55_per_tensor",
        ),
        pytest.param(
            CortexM.M4,
            True,
            exir_ops.edge.cortex_m.quantized_linear_s4.default,
            id="m4",
        ),
    ],
)
def test_dialect_linear_a8w4_unaligned_rows_use_int8_kernels_on_mve(
    cpu, per_channel, expected
):
    # Rows of 37 values: unaligned VLD2 on MVE.
    case = McuTestCase(
        model=CortexMLinearPerChannel(37, 13, bias=True),
        example_inputs=(ramp_tensor(-2, 2, (2, 37)),),
    )
    config = A8W4_PER_CHANNEL_CONFIG if per_channel else _per_tensor_a8w4_config()
    tester = _lower_linear_per_channel(case, CortexMTargetConfig(cpu=cpu), config)
    tester.run_method_and_compare_outputs(inputs=case.example_inputs, qtol=1)

    if expected == exir_ops.edge.cortex_m.quantized_linear_s4.default:
        _s4_linear_weights(tester)
        return
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    [linear] = _linear_nodes(tester)
    assert linear.target == expected
    assert get_param_tensor(program, linear.args[1]).abs().max() == 7


@parametrize("test_case", a8w4_linear_test_cases)
def test_implementation_linear_a8w4(test_case, cortex_m_target):
    tester = _lower_linear_per_channel(
        test_case, cortex_m_target, A8W4_PER_CHANNEL_CONFIG
    )
    tester.to_executorch().serialize()
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs, qtol=1)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize(
    "packed_shape",
    [pytest.param((5,), id="wrong_byte_count"), pytest.param((2, 3), id="not_1d")],
)
def test_quantized_linear_s4_rejects_invalid_weights(fake, packed_shape):
    # 3 output features over 4 input features take 6 bytes.
    args = (
        torch.ones((2, 4), dtype=torch.int8),
        torch.zeros(packed_shape, dtype=torch.int8),
        None,
        0,
        0,
        torch.full((3,), 1 << 30, dtype=torch.int32),
        torch.ones(3, dtype=torch.int32),
        127,
        -128,
    )
    with FakeTensorMode(allow_non_fake_inputs=True) if fake else nullcontext():
        with pytest.raises(ValueError, match="two per byte"):
            torch.ops.cortex_m.quantized_linear_s4.default(*args)


def test_quantized_linear_s4_matches_unpacked_reference():
    weights = torch.tensor([[1, -7, 3], [7, -2, 0]], dtype=torch.int8)
    args = (
        torch.tensor([[10, -20, 30]], dtype=torch.int8),
        pack_int4(weights),
        torch.tensor([5, -5], dtype=torch.int32),
        3,
        -1,
        torch.full((2,), 1 << 30, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        127,
        -128,
    )
    expected = torch.ops.cortex_m.quantized_linear_per_channel.default(
        args[0], weights, args[2], None, args[3], 0, *args[4:]
    )
    assert torch.equal(torch.ops.cortex_m.quantized_linear_s4.default(*args), expected)
