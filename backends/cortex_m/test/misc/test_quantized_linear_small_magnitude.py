# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Regression test for the cortex_m::quantized_linear bias/kernel_sum bug.

CMSIS-NN's `arm_fully_connected_s8` has three runtime paths, gated by
compile-time ARM_MATH_MVEI / ARM_MATH_DSP:

* MVE: reads ctx.buf (precomputed kernel_sum that includes bias plus
  input_offset x sum(weight)), ignores the bias argument.
* DSP / scalar: read the bias argument directly, ignore ctx.buf.

`ConvertToCortexMPass._get_linear_replacement` selects which input format
to emit based on `CortexMTargetConfig.backend`. Before the fix, the pass
unconditionally emitted kernel_sum + None-bias, which on a non-MVE build
silently dropped both the bias and the input-offset term. The bug only
showed up when those terms dominated the int32 accumulator, i.e. on
small-magnitude inputs.

Coverage:

* `test_dialect_small_magnitude_linear` runs each ISA through the Python
  op impl and checks that bias=True and bias=False variants both round
  to the same int8 outputs as the float reference.
* `test_aot_graph_shape_small_magnitude_linear` inspects the post-pass
  graph and asserts the bias/kernel_sum arg positions match the ISA
  convention -- this is the direct regression check.
* `test_implementation_small_magnitude_linear` runs the bias=True case
  through the default (M55, MVE) build path so the impl test exercises
  the kernel_sum codepath in simulation.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as exir_ops

torch.manual_seed(0)


class _SmallMagnitudeLinear(nn.Module):
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
        self.fc = nn.Linear(512, 10, bias=bias)

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


def _make_input():
    return torch.rand(1, 512) * 0.002


_calibration_samples = [(_make_input(),) for _ in range(8)]


@dataclass(frozen=True)
class _IsaVariant:
    case: McuTestCase
    target_config: CortexMTargetConfig
    uses_kernel_sum: bool


def _variant(model_cls, cpu: CortexM, uses_kernel_sum: bool) -> _IsaVariant:
    return _IsaVariant(
        case=McuTestCase(
            model=model_cls().eval(),
            example_inputs=lambda: (_make_input(),),
        ),
        target_config=CortexMTargetConfig(cpu=cpu),
        uses_kernel_sum=uses_kernel_sum,
    )


# bias=True covers the regression directly (the bug dropped the bias
# term); bias=False covers the symmetric case where only the
# input-offset term is missing on the non-MVE paths.
test_variants = {
    "mve_bias": _variant(_SmallMagnitudeLinear, CortexM.M55, uses_kernel_sum=True),
    "dsp_bias": _variant(_SmallMagnitudeLinear, CortexM.M4, uses_kernel_sum=False),
    "scalar_bias": _variant(_SmallMagnitudeLinear, CortexM.M0PLUS, uses_kernel_sum=False),
    "mve_nobias": _variant(_SmallMagnitudeLinearNoBias, CortexM.M55, uses_kernel_sum=True),
    "dsp_nobias": _variant(_SmallMagnitudeLinearNoBias, CortexM.M4, uses_kernel_sum=False),
    "scalar_nobias": _variant(
        _SmallMagnitudeLinearNoBias, CortexM.M0PLUS, uses_kernel_sum=False
    ),
}


@parametrize("variant", test_variants)
def test_dialect_small_magnitude_linear(variant: _IsaVariant):
    tester = CortexMTester(
        variant.case.model,
        variant.case.get_example_inputs(),
        target_config=variant.target_config,
    )
    tester.test_dialect(
        ops_before_transforms=variant.case.model.ops_before_transforms,
        ops_after_transforms=variant.case.model.ops_after_transforms,
        qtol=1,
        calibration_samples=_calibration_samples,
    )


@parametrize("variant", test_variants)
def test_aot_graph_shape_small_magnitude_linear(variant: _IsaVariant):
    """Assert the post-pass node args match the ISA's CMSIS-NN convention."""
    tester = CortexMTester(
        variant.case.model,
        variant.case.get_example_inputs(),
        target_config=variant.target_config,
    )
    tester.quantize(None)
    tester.export()
    tester.to_edge()
    tester.run_passes()

    module = tester.get_artifact(StageType.RUN_PASSES).exported_program().module()
    linear_target = exir_ops.edge.cortex_m.quantized_linear.default
    linear_nodes = [
        n for n in module.graph.nodes if n.op == "call_function" and n.target == linear_target
    ]
    assert len(linear_nodes) == 1, f"expected one quantized_linear node, got {len(linear_nodes)}"
    bias_arg, kernel_sum_arg = linear_nodes[0].args[2], linear_nodes[0].args[3]

    if variant.uses_kernel_sum:
        assert bias_arg is None, "MVE path must not pass bias (CMSIS-NN ignores it)"
        assert kernel_sum_arg is not None, "MVE path requires precomputed kernel_sum"
    else:
        assert kernel_sum_arg is None, "non-MVE path must not pass kernel_sum"
        # bias is allowed to be None only if the source nn.Linear had bias=False.
        expects_bias = variant.case.model.fc.bias is not None
        if expects_bias:
            assert bias_arg is not None, "non-MVE path with bias must forward bias to CMSIS-NN"


def test_implementation_small_magnitude_linear():
    """Exercise the MVE kernel_sum codepath via the default M55 simulator build."""
    case = McuTestCase(
        model=_SmallMagnitudeLinear().eval(),
        example_inputs=lambda: (_make_input(),),
    )
    tester = CortexMTester(case.model, case.get_example_inputs())
    tester.test_implementation(qtol=1, calibration_samples=_calibration_samples)
