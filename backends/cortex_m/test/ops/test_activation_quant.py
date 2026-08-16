# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)


# A single per-op `ops_after_transforms` shape is enough: every supported
# activation lowers to exactly one cortex_m.quantized_activation, with the
# AoT LUT stored as a constant placeholder and a single quant/dequant pair
# at the graph boundary.
_OPS_BEFORE = {
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
}
_OPS_AFTER = {
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_activation_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
}


class _Sigmoid(torch.nn.Module):
    ops_before_transforms = {
        **_OPS_BEFORE,
        "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1,
    }
    ops_after_transforms = _OPS_AFTER

    def forward(self, x):
        return torch.sigmoid(x)


# nn.Sigmoid and nn.Tanh take no `inplace` argument, so the tensor method is
# the only way to reach aten.sigmoid_ / aten.tanh_ from Python.
class _SigmoidInplace(torch.nn.Module):
    ops_before_transforms = {
        **_OPS_BEFORE,
        "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 1,
    }
    ops_after_transforms = _OPS_AFTER

    def forward(self, x):
        return x.sigmoid_()


class _Tanh(torch.nn.Module):
    ops_before_transforms = {
        **_OPS_BEFORE,
        "executorch_exir_dialects_edge__ops_aten_tanh_default": 1,
    }
    ops_after_transforms = _OPS_AFTER

    def forward(self, x):
        return torch.tanh(x)


class _TanhInplace(torch.nn.Module):
    ops_before_transforms = {
        **_OPS_BEFORE,
        "executorch_exir_dialects_edge__ops_aten_tanh_default": 1,
    }
    ops_after_transforms = _OPS_AFTER

    def forward(self, x):
        return x.tanh_()


class _SiLU(torch.nn.Module):
    ops_before_transforms = {
        **_OPS_BEFORE,
        "executorch_exir_dialects_edge__ops_aten_silu_default": 1,
    }
    ops_after_transforms = _OPS_AFTER

    def forward(self, x):
        return torch.nn.functional.silu(x)


class _SiLUInplace(torch.nn.Module):
    ops_before_transforms = {
        **_OPS_BEFORE,
        "executorch_exir_dialects_edge__ops_aten_silu_default": 1,
    }
    ops_after_transforms = _OPS_AFTER

    def __init__(self):
        super().__init__()
        self.silu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(x)


class _ConvSiLUInplace(torch.nn.Module):
    """The shape a real model has: the activation consumes a convolution
    output, so the conv is matched first by the per-channel quantizer and the
    activation only afterwards.
    """

    # No _OPS_BEFORE here: the convolution brings its own weight quant/dequant,
    # so the boundary counts the other cases share do not apply.
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_silu_default": 1,
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_activation_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_aten_silu_default": 0,
    }

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3, padding=1)
        self.silu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.conv(x))


class _GELU(torch.nn.Module):
    ops_before_transforms = {
        **_OPS_BEFORE,
        "executorch_exir_dialects_edge__ops_aten_gelu_default": 1,
    }
    ops_after_transforms = {
        **_OPS_AFTER,
        "executorch_exir_dialects_edge__ops_aten_gelu_default": 0,
    }

    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()  # default: exact / erf

    def forward(self, x):
        return self.gelu(x)


import torch as _torch


def _zero_input(shape):
    return _torch.zeros(shape, dtype=_torch.float32)


# Wide-magnitude inputs exercise the `max(-128, min(127, q_out))` clamp inside
# build_activation_lut; shifted-ramp inputs push the quantizer to pick a
# non-zero `input_zp`, exercising the `(q - input_zp) * input_scale` term in
# the LUT formula; all-zero inputs pin down the lut entry at `input_zp + 128`.
test_cases = {
    "sigmoid_rank1": McuTestCase(
        model=_Sigmoid(),
        example_inputs=(ramp_tensor(-6, 6, (16,)),),
    ),
    "sigmoid_rank4": McuTestCase(
        model=_Sigmoid(),
        example_inputs=(ramp_tensor(-4, 4, (1, 8, 4, 4)),),
    ),
    "sigmoid_saturating": McuTestCase(
        model=_Sigmoid(),
        example_inputs=(ramp_tensor(-50, 50, (32,)),),
    ),
    "sigmoid_asymmetric_zp": McuTestCase(
        model=_Sigmoid(),
        example_inputs=(ramp_tensor(-1, 9, (16,)),),
    ),
    "sigmoid_zero": McuTestCase(
        model=_Sigmoid(),
        example_inputs=(_zero_input((16,)),),
    ),
    # These three activate the placeholder itself, so calibration rewrites the
    # input tensor. Building it per call keeps one case from feeding the next;
    # within a case both sides still see the rewritten tensor, which narrows the
    # compared range. That is fine here -- they exist to prove the in-place
    # spelling gets annotated, and the functional siblings above already cover
    # the LUT over its full range -- but do not read them as range coverage.
    "sigmoid_inplace": McuTestCase(
        model=_SigmoidInplace(),
        example_inputs=lambda: (ramp_tensor(-4, 4, (1, 8, 4, 4)),),
    ),
    "tanh_rank1": McuTestCase(
        model=_Tanh(),
        example_inputs=(ramp_tensor(-3, 3, (16,)),),
    ),
    "tanh_rank3": McuTestCase(
        model=_Tanh(),
        example_inputs=(ramp_tensor(-2, 2, (1, 4, 16)),),
    ),
    "tanh_saturating": McuTestCase(
        model=_Tanh(),
        example_inputs=(ramp_tensor(-30, 30, (32,)),),
    ),
    "tanh_asymmetric_zp": McuTestCase(
        model=_Tanh(),
        example_inputs=(ramp_tensor(-1, 5, (16,)),),
    ),
    "tanh_zero": McuTestCase(
        model=_Tanh(),
        example_inputs=(_zero_input((16,)),),
    ),
    "tanh_inplace": McuTestCase(
        model=_TanhInplace(),
        example_inputs=lambda: (ramp_tensor(-2, 2, (1, 8, 4, 4)),),
    ),
    "silu_rank1": McuTestCase(
        model=_SiLU(),
        example_inputs=(ramp_tensor(-6, 6, (16,)),),
    ),
    "silu_rank4": McuTestCase(
        model=_SiLU(),
        example_inputs=(ramp_tensor(-4, 4, (1, 8, 4, 4)),),
    ),
    "silu_saturating": McuTestCase(
        model=_SiLU(),
        example_inputs=(ramp_tensor(-50, 50, (32,)),),
    ),
    "silu_asymmetric_zp": McuTestCase(
        model=_SiLU(),
        example_inputs=(ramp_tensor(-1, 9, (16,)),),
    ),
    "silu_zero": McuTestCase(
        model=_SiLU(),
        example_inputs=(_zero_input((16,)),),
    ),
    "silu_inplace": McuTestCase(
        model=_SiLUInplace(),
        example_inputs=lambda: (ramp_tensor(-4, 4, (1, 8, 4, 4)),),
    ),
    "conv_silu_inplace": McuTestCase(
        model=_ConvSiLUInplace(),
        example_inputs=lambda: (
            ramp_tensor(-4, 4, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "gelu_rank1": McuTestCase(
        model=_GELU(),
        example_inputs=(ramp_tensor(-6, 6, (16,)),),
    ),
    "gelu_rank4": McuTestCase(
        model=_GELU(),
        example_inputs=(ramp_tensor(-4, 4, (1, 8, 4, 4)),),
    ),
    "gelu_saturating": McuTestCase(
        model=_GELU(),
        example_inputs=(ramp_tensor(-50, 50, (32,)),),
    ),
    "gelu_asymmetric_zp": McuTestCase(
        model=_GELU(),
        example_inputs=(ramp_tensor(-1, 9, (16,)),),
    ),
    "gelu_zero": McuTestCase(
        model=_GELU(),
        example_inputs=(_zero_input((16,)),),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_quantized_activation(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


@parametrize("test_case", test_cases)
def test_implementation_quantized_activation(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(qtol=1)
