# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)
from executorch.backends.test.harness.stages import StageType
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
def test_dialect_linear(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


@parametrize("test_case", test_cases)
def test_implementation_linear(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
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
# An additional implementation test drives the default M55 MVE build path
# through the simulator.
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


def _small_magnitude_variant(
    model_cls, cpu: CortexM, uses_kernel_sum: bool
) -> _SmallMagnitudeVariant:
    return _SmallMagnitudeVariant(
        case=McuTestCase(
            model=model_cls().eval(),
            example_inputs=lambda: (_small_magnitude_input(),),
        ),
        target_config=CortexMTargetConfig(cpu=cpu),
        uses_kernel_sum=uses_kernel_sum,
    )


# bias=True covers the regression directly (the bug dropped the bias term);
# bias=False covers the symmetric case where only the input-offset term is
# missing on the non-MVE paths.
small_magnitude_variants = {
    "mve_bias": _small_magnitude_variant(
        _SmallMagnitudeLinear, CortexM.M55, uses_kernel_sum=True
    ),
    "dsp_bias": _small_magnitude_variant(
        _SmallMagnitudeLinear, CortexM.M4, uses_kernel_sum=False
    ),
    "scalar_bias": _small_magnitude_variant(
        _SmallMagnitudeLinear, CortexM.M0PLUS, uses_kernel_sum=False
    ),
    "mve_nobias": _small_magnitude_variant(
        _SmallMagnitudeLinearNoBias, CortexM.M55, uses_kernel_sum=True
    ),
    "dsp_nobias": _small_magnitude_variant(
        _SmallMagnitudeLinearNoBias, CortexM.M4, uses_kernel_sum=False
    ),
    "scalar_nobias": _small_magnitude_variant(
        _SmallMagnitudeLinearNoBias, CortexM.M0PLUS, uses_kernel_sum=False
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
        assert isinstance(variant.case.model, _SmallMagnitudeLinear)
        if variant.case.model.fc.bias is None:
            assert bias_arg is None


def test_implementation_linear_small_magnitude():
    """Exercise the MVE kernel_sum codepath via the default M55 simulator build."""
    case = McuTestCase(
        model=_SmallMagnitudeLinear().eval(),
        example_inputs=lambda: (_small_magnitude_input(),),
    )
    tester = CortexMTester(case.model, case.get_example_inputs())
    tester.test_implementation(qtol=1, calibration_samples=_small_magnitude_calibration)
