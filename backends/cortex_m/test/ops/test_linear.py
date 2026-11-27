# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass

import pytest
import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.quantizer.quantization_configs import (
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
from executorch.backends.cortex_m.utils import requantize_cmsis
from executorch.backends.test.harness.stages import Quantize, StageType
from executorch.backends.transforms.utils import get_param_tensor
from executorch.exir.dialects._ops import ops as exir_ops


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
        requantize_tensor = get_param_tensor(program, requantize_arg)
        assert requantize_tensor.shape == (out_features,)
        assert requantize_tensor.dtype == torch.int32
    # Per-channel scales make every weight row reach the end of the int8 range.
    weights = get_param_tensor(program, linear.args[1]).to(torch.int32)
    assert set(weights.abs().amax(dim=1).tolist()) <= {127, 128}


@parametrize("test_case", per_channel_linear_test_cases)
def test_implementation_linear_per_channel(test_case, cortex_m_target):
    tester = _lower_linear_per_channel(test_case, cortex_m_target)
    tester.to_executorch().serialize()
    tester.run_method_and_compare_outputs(inputs=test_case.example_inputs, qtol=1)


def test_requantize_cmsis_per_channel():
    # Channel 0 scales by 0.5 (shift 0), channel 1 by 0.25 (shift -1).
    acc = torch.tensor([[4, 5], [5, -5], [-5, -7]], dtype=torch.int32)
    multipliers = torch.tensor([1 << 30, 1 << 30], dtype=torch.int32)
    shifts = torch.tensor([0, -1], dtype=torch.int32)
    expected = torch.tensor([[2, 2], [3, -1], [-2, -2]], dtype=torch.int32)
    assert torch.equal(requantize_cmsis(acc, multipliers, shifts), expected)


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
