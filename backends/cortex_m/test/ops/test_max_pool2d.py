# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.test.common import parametrize, xfail_type
from executorch.backends.cortex_m.passes.passes_utils import ceil_mode_is_redundant
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


class CortexMMaxPool2dFunctional(torch.nn.Module):
    ops_before_transforms = CortexMMaxPool2d.ops_before_transforms
    ops_after_transforms = CortexMMaxPool2d.ops_after_transforms

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(x, **self.kwargs)


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
    # SqueezeNet's pool: ceil_mode is set but 11 - 3 divides by 2, so rounding
    # up picks the same output size.
    "maxpool_3x3_s2_ceil_redundant": McuTestCase(
        CortexMMaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        (ramp_tensor(-20, 20, (1, 1, 11, 11)),),
    ),
    # An odd stride, so the padding term decides the answer, over a 169-element
    # plane -- past the 127 that aten's channels-last int8 pool accepts, so
    # this one could not have been written as a fallback case.
    "maxpool_3x3_s3_pad1_ceil_redundant": McuTestCase(
        CortexMMaxPool2d(kernel_size=3, stride=3, padding=1, ceil_mode=True),
        ((torch.randn(1, 8, 13, 13) * 30).to(memory_format=torch.channels_last),),
    ),
    # nn.MaxPool2d always fills in a stride; the functional spelling leaves the
    # schema default of [] in the graph, which only appears once a later
    # argument is named.
    "maxpool_3x3_default_stride_ceil": McuTestCase(
        CortexMMaxPool2dFunctional(kernel_size=3, ceil_mode=True),
        (ramp_tensor(-20, 20, (1, 1, 12, 12)),),
    ),
}


fallback_test_cases = {
    "maxpool_3x3_s2_pad1_ceil": McuTestCase(
        CortexMMaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
        (ramp_tensor(-10, 10, (1, 1, 4, 4)),),
    ),
    # The height divides but the width does not, so the ceiling still adds a
    # column and the whole pool has to decline.
    "maxpool_3x3_s2_ceil_one_axis": McuTestCase(
        CortexMMaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        (ramp_tensor(-20, 20, (1, 1, 11, 12)),),
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


def test_ceil_mode_redundancy_reads_every_term():
    """Each end-to-end case exercises one configuration, leaving every term it
    does not vary unconstrained."""
    square = torch.Size([11, 11])
    assert ceil_mode_is_redundant(square, (3, 3), (2, 2), (0, 0), (1, 1))

    # Height and width are read as themselves: with a per-axis stride, only one
    # of the two orders divides.
    assert not ceil_mode_is_redundant(
        torch.Size([11, 9]), (3, 3), (2, 4), (0, 0), (1, 1)
    )
    assert ceil_mode_is_redundant(torch.Size([9, 11]), (3, 3), (2, 4), (0, 0), (1, 1))

    # Padding and dilation both move the span, which only an odd stride notices.
    ten = torch.Size([10, 10])
    assert not ceil_mode_is_redundant(ten, (3, 3), (3, 3), (0, 0), (1, 1))
    assert ceil_mode_is_redundant(ten, (3, 3), (3, 3), (1, 1), (1, 1))
    assert not ceil_mode_is_redundant(ten, (3, 3), (3, 3), (1, 1), (2, 2))


class Identity(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


def test_ceil_mode_redundancy_declines_a_symbolic_shape():
    """A symbolic dimension divides symbolically rather than raising, so
    without the guard the predicate would answer for whichever size happened to
    be traced and bake that into the kernel. A max pool cannot supply one --
    export rejects a dynamic spatial dim on it -- so borrow a shape.
    """
    exported = torch.export.export(
        Identity().eval(),
        (torch.randn(1, 1, 11, 11),),
        dynamic_shapes=({2: torch.export.Dim("height", min=4, max=32)},),
    )
    (placeholder,) = [n for n in exported.graph.nodes if n.op == "placeholder"]
    shape = placeholder.meta["val"].shape[-2:]
    assert not all(isinstance(n, int) for n in shape), shape
    assert not ceil_mode_is_redundant(shape, (3, 3), (2, 2), (0, 0), (1, 1))


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
