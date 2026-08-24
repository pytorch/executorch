# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.test.common import parametrize, xfail_type
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)


class CortexMMaxPool2d(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_max_pool2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class CortexMMaxPool2dPermutedView(torch.nn.Module):
    """A single-channel NHWC image permuted to NCHW.

    With C == 1 the channels-last strides also satisfy plain contiguity, so
    .contiguous() hands the reference back the very tensor it was given while
    aten keeps dispatching on the memory-format hint. This case is what pins
    the .to(memory_format=...) in quantized_max_pool2d_impl: it fails at the
    dialect stage if that is written as .contiguous().
    """

    # The permute brings its own quant/dequant pair, so only the pool itself
    # is pinned here.
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 1,
    }
    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_max_pool2d_default": 1,
        "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 0,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x.permute(0, 3, 1, 2))


class CortexMMaxPool2dIndices(torch.nn.Module):
    ops_before_transforms = CortexMMaxPool2d.ops_before_transforms
    ops_after_transforms = CortexMMaxPool2d.ops_after_transforms

    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs["return_indices"] = True
        self.pool = torch.nn.MaxPool2d(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)[1]


test_cases = {
    "maxpool_2x2": McuTestCase(
        CortexMMaxPool2d(kernel_size=2, stride=2),
        (ramp_tensor(-50, 50, (1, 1, 6, 6)),),
    ),
    "maxpool_3x3_s1": McuTestCase(
        CortexMMaxPool2d(kernel_size=3, stride=1, padding=1),
        (ramp_tensor(-20, 20, (1, 1, 5, 5)),),
    ),
    "maxpool_2x2_pad1": McuTestCase(
        CortexMMaxPool2d(kernel_size=2, stride=2, padding=1),
        (ramp_tensor(-30, 30, (1, 1, 7, 7)),),
    ),
    "maxpool_3x3_s2_pad1": McuTestCase(
        CortexMMaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        (ramp_tensor(-16, 16, (1, 1, 6, 6)),),
    ),
    "maxpool_2x2_indices": McuTestCase(
        CortexMMaxPool2dIndices(kernel_size=2, stride=2),
        (ramp_tensor(-50, 50, (1, 1, 6, 6)),),
    ),
    # 576 spatial elements (24x24), past the 127 that aten's channels-last int8
    # max_pool2d accepts; the reference pools a contiguous copy to avoid it.
    # randn rather than a ramp: a ramp over this many elements puts a whole
    # pooling window inside one int8 code, so it cannot tell max from min.
    "maxpool_2x2_large_channels_last": McuTestCase(
        CortexMMaxPool2d(kernel_size=2, stride=2),
        ((torch.randn(1, 64, 24, 24) * 30).to(memory_format=torch.channels_last),),
    ),
    "maxpool_2x2_single_channel_view": McuTestCase(
        CortexMMaxPool2dPermutedView(kernel_size=2, stride=2),
        ((torch.randn(1, 24, 24, 1) * 30),),
    ),
}


fallback_test_cases = {
    "maxpool_3x3_s2_pad1_ceil": McuTestCase(
        CortexMMaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        (ramp_tensor(-10, 10, (1, 1, 4, 4)),),
    ),
    "maxpool_dilation": McuTestCase(
        CortexMMaxPool2d(kernel_size=2, stride=1, padding=0, dilation=2),
        (ramp_tensor(-25, 25, (1, 1, 6, 6)),),
    ),
}

xfails_max_pool2d: dict[str, xfail_type] = {
    "maxpool_2x2_indices": (
        "Indices output not supported; quantizer does not handle getitem on max_pool2d_with_indices.",
        Exception,
    ),
}


@parametrize("test_case", test_cases, xfails=xfails_max_pool2d)
def test_dialect_max_pool2d(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


@parametrize("test_case", fallback_test_cases)
def test_dialect_max_pool2d_fallback(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        {
            "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 1,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
            "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        },
        {
            "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
            "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
            "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 1,
        },
        qtol=1,
    )


@parametrize("test_case", fallback_test_cases)
def test_executorch_max_pool2d_fallback(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.quantize().export().to_edge().run_passes().to_executorch()


@parametrize("test_case", fallback_test_cases)
def test_implementation_max_pool2d_fallback(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(qtol=1)


@parametrize("test_case", test_cases, xfails=xfails_max_pool2d)
def test_implementation_max_pool2d(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(qtol=1)
