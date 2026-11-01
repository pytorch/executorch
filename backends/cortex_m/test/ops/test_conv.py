# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from contextlib import nullcontext

import pytest
import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a8w4_quantization_config,
)
from executorch.backends.arm.test.common import parametrize, xfail_type
from executorch.backends.cortex_m.library import cmsis_nn
from executorch.backends.cortex_m.passes.scratch_buffer_sizes import (
    _use_generic_s4_kernel,
)
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    A8W4_PER_CHANNEL_CONFIG,
    CortexMQuantizationConfig,
)
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.cortex_m.test.tester import (
    CortexMRunPasses,
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)
from executorch.backends.cortex_m.utils import pack_int4, unpack_int4
from executorch.backends.test.harness.stages import Quantize, StageType
from executorch.backends.transforms.utils import get_param_tensor
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


class CortexMConv1D(torch.nn.Module):
    ops_before_transforms: dict[str, int] = {}
    ops_after_transforms: dict[str, int] = {}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv1d(*args, **kwargs, bias=False)

    def forward(self, x):
        return self.conv(x)


class CortexMConv2D(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(*args, **kwargs, bias=False)
        self.conv.weight.data.fill_(1.0)

    def forward(self, x):
        return self.conv(x)


class CortexMConv2DBias(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 2,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(*args, **kwargs, bias=True)

    def forward(self, x):

        return self.conv(x)


class CortexMConv3D(torch.nn.Module):
    ops_before_transforms: dict[str, int] = {}
    ops_after_transforms: dict[str, int] = {}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv3d(*args, **kwargs, bias=False)
        self.conv.weight.data.fill_(2.0)

    def forward(self, x):
        return self.conv(x)


class CortexMConv2Dx3(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 3,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 4,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 4,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 3,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_default": 3,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(16, 8, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class CortexMDepthwiseConv2D(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_depthwise_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(*args, **kwargs, bias=False)

    def forward(self, x):
        return self.conv(x)


class CortexMDepthwiseConv2DBias(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_convolution_default": 1,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 2,
        "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_channel_default": 2,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_depthwise_conv2d_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 1,
        "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(*args, **kwargs, bias=True)

    def forward(self, x):
        return self.conv(x)


# in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
test_cases = {
    "conv2d": McuTestCase(
        model=CortexMConv2D(2, 4, 3),
        example_inputs=(
            ramp_tensor(1, 5, (1, 2, 5, 5)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_stride": McuTestCase(
        model=CortexMConv2D(3, 4, (1, 2), stride=2),
        example_inputs=(
            ramp_tensor(-100, 10, (3, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_padding": McuTestCase(
        model=CortexMConv2D(3, 2, 3, padding=(4, 1)),
        example_inputs=(
            ramp_tensor(0, 1, (2, 3, 5, 5)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_dilation": McuTestCase(
        model=CortexMConv2D(1, 4, 3, dilation=(2, 2)),
        example_inputs=(
            ramp_tensor(0, 10, (3, 1, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    # A bias-less grouped convolution is what needs a bias synthesized for it, so
    # keep this case bias-less, keep groups > 1, and keep the reference output
    # away from saturation: the previous 1x1-output version summed an
    # all-positive ramp straight to qmax, where a wrong result is
    # indistinguishable from the correct one.
    "conv2d_groups": McuTestCase(
        model=CortexMConv2D(4, 4, 3, padding=1, groups=2),
        example_inputs=(
            ramp_tensor(-5, 5, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_groups_bias": McuTestCase(
        model=CortexMConv2DBias(4, 4, 3, padding=1, groups=2),
        example_inputs=(
            ramp_tensor(-5, 5, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_bias_ch_out_1": McuTestCase(
        model=CortexMConv2DBias(5, 1, 1),
        example_inputs=(
            ramp_tensor(0, 10, (2, 5, 3, 3)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_bias_ch_out_4": McuTestCase(
        model=CortexMConv2DBias(5, 4, (1, 2)),
        example_inputs=(
            ramp_tensor(-3, 3, (2, 5, 10, 10)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_nchw": McuTestCase(
        model=CortexMConv2D(5, 5, 1),
        example_inputs=(ramp_tensor(0, 10, (1, 5, 8, 8)),),
    ),
    "conv1d": McuTestCase(
        model=CortexMConv1D(1, 1, 1),
        example_inputs=(ramp_tensor(0, 10, (1, 3, 2)),),
    ),
    "conv3d": McuTestCase(
        model=CortexMConv3D(1, 1, 1),
        example_inputs=(
            ramp_tensor(-1000, 1000, (2, 1, 3, 3, 3)).to(
                memory_format=torch.channels_last_3d
            ),
        ),
    ),
    "conv2d_x3": McuTestCase(
        model=CortexMConv2Dx3(),
        example_inputs=(
            ramp_tensor(0, 10, (1, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    # Depthwise convolution tests (groups == in_channels)
    "depthwise_conv2d": McuTestCase(
        model=CortexMDepthwiseConv2D(4, 4, 3, groups=4),
        example_inputs=(
            ramp_tensor(1, 5, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "depthwise_conv2d_multiplier": McuTestCase(
        model=CortexMDepthwiseConv2D(3, 6, 3, groups=3),
        example_inputs=(
            ramp_tensor(0, 10, (1, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "depthwise_conv2d_stride": McuTestCase(
        model=CortexMDepthwiseConv2D(4, 4, 3, stride=2, groups=4),
        example_inputs=(
            ramp_tensor(-50, 50, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "depthwise_conv2d_padding": McuTestCase(
        model=CortexMDepthwiseConv2D(2, 2, 5, padding=2, groups=2),
        example_inputs=(
            ramp_tensor(0, 1, (1, 2, 5, 5)).to(memory_format=torch.channels_last),
        ),
    ),
    "depthwise_conv2d_bias": McuTestCase(
        model=CortexMDepthwiseConv2DBias(3, 3, 3, padding=1, groups=3),
        example_inputs=(
            ramp_tensor(-10, 10, (1, 3, 6, 6)).to(memory_format=torch.channels_last),
        ),
    ),
    "depthwise_conv2d_stride_padding_bias": McuTestCase(
        model=CortexMDepthwiseConv2DBias(4, 4, 3, stride=2, padding=1, groups=4),
        example_inputs=(
            ramp_tensor(0, 5, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "depthwise_conv2d_1x1": McuTestCase(
        model=CortexMDepthwiseConv2D(4, 8, 1, groups=4),
        example_inputs=(
            ramp_tensor(0, 10, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "depthwise_conv2d_multiplier_4": McuTestCase(
        model=CortexMDepthwiseConv2D(2, 8, 3, groups=2),
        example_inputs=(
            ramp_tensor(0, 10, (1, 2, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "depthwise_conv2d_asymmetric_kernel": McuTestCase(
        model=CortexMDepthwiseConv2D(4, 4, (1, 3), groups=4),
        example_inputs=(
            ramp_tensor(0, 10, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "depthwise_conv2d_asymmetric_stride": McuTestCase(
        model=CortexMDepthwiseConv2D(3, 3, 3, stride=(2, 1), padding=(1, 0), groups=3),
        example_inputs=(
            ramp_tensor(0, 10, (1, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "depthwise_conv2d_5x5": McuTestCase(
        model=CortexMDepthwiseConv2D(4, 4, 5, padding=2, groups=4),
        example_inputs=(
            ramp_tensor(0, 10, (1, 4, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "depthwise_conv2d_minimal_spatial": McuTestCase(
        model=CortexMDepthwiseConv2D(
            2, 2, 3, padding=1, groups=2
        ),  # 3x3 kernel on 1x1 input
        example_inputs=(
            ramp_tensor(0, 10, (1, 2, 1, 1)).to(memory_format=torch.channels_last),
        ),
    ),
    # Single-channel depthwise convolution tests (in_channels == groups == 1)
    "depthwise_conv2d_single_channel": McuTestCase(
        model=CortexMDepthwiseConv2D(1, 1, 3, groups=1),
        example_inputs=(
            ramp_tensor(0, 10, (1, 1, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "depthwise_conv2d_single_channel_multiplier": McuTestCase(
        model=CortexMDepthwiseConv2D(1, 3, 3, groups=1),
        example_inputs=(
            ramp_tensor(0, 10, (1, 1, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
}


xfails_dialect: dict[str, xfail_type] = {
    "conv2d_dilation": "NotImplementedError: 'slow_conv_dilated<>' not implemented for 'Int'",
    "conv1d": "Currently not supported.",
    "conv2d_nchw": "Currently not supported.",
}


@parametrize("test_case", test_cases, xfails=xfails_dialect)
def test_dialect_conv2d(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=1,
    )


xfails_implementation: dict[str, xfail_type] = {
    "conv1d": "Currently not supported.",
    "conv3d": "Currently not supported.",
}


@parametrize("test_case", test_cases, xfails=xfails_implementation)
def test_implementation_conv2d(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(qtol=2)


@parametrize(
    "test_case",
    {name: test_cases[name] for name in ("conv2d_groups", "depthwise_conv2d")},
)
def test_grouped_conv2d_bias_is_populated(test_case, cortex_m_target):
    """Assert a bias-less grouped convolution reaches the kernel with a bias.

    The numeric cases only cover this on the FVP leg and only while their
    reference output stays clear of saturation, and the depthwise kernel
    offsets the bias by whole channel blocks, so it takes hundreds of channels
    before a missing bias shows up at all -- far more than any case here.
    """
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.quantize()
    tester.export()
    tester.to_edge()
    tester.run_passes()

    module = tester.get_artifact(StageType.RUN_PASSES).exported_program().module()
    grouped_convs = (
        exir_ops.edge.cortex_m.quantized_conv2d.default,
        exir_ops.edge.cortex_m.quantized_depthwise_conv2d.default,
    )
    [conv_node] = [
        n
        for n in module.graph.nodes
        if n.op == "call_function" and n.target in grouped_convs
    ]
    assert conv_node.args[2] is not None


@pytest.mark.parametrize(
    "backend", [cmsis_nn.Backend.SCALAR, cmsis_nn.Backend.DSP, cmsis_nn.Backend.MVE]
)
@pytest.mark.parametrize("out_channels", [1, 8, 9, 64])
def test_dialect_single_channel_dispatch(backend, out_channels):
    model = torch.nn.Conv2d(1, out_channels, 3, bias=False).eval()
    inputs = (torch.randn(1, 1, 8, 8).to(memory_format=torch.channels_last),)
    config = CortexMTargetConfig(cpu=CortexM.M55, isa=backend)
    tester = CortexMTester(model, inputs, target_config=config)
    tester.quantize().export().to_edge().run_passes()
    tester.run_method_and_compare_outputs(inputs=inputs, qtol=1)
    graph = tester.get_artifact(StageType.RUN_PASSES).exported_program().graph
    regular = backend == cmsis_nn.Backend.MVE and out_channels > 8
    expected = (
        exir_ops.edge.cortex_m.quantized_conv2d.default
        if regular
        else exir_ops.edge.cortex_m.quantized_depthwise_conv2d.default
    )
    assert sum(node.target == expected for node in graph.nodes) == 1


@pytest.mark.parametrize("out_channels", [1, 8, 9, 64])
def test_implementation_single_channel_dispatch(out_channels, cortex_m_target):
    model = torch.nn.Conv2d(1, out_channels, 3, bias=False).eval()
    inputs = (torch.randn(1, 1, 8, 8).to(memory_format=torch.channels_last),)
    CortexMTester(model, inputs, target_config=cortex_m_target).test_implementation(
        qtol=1
    )


def test_pack_int4_nibble_order():
    values = torch.tensor([1, -2, 3], dtype=torch.int8)
    packed = pack_int4(values)
    assert torch.equal(
        packed, torch.tensor([0xE1, 0x03], dtype=torch.uint8).view(torch.int8)
    )
    assert torch.equal(unpack_int4(packed, [3]), values)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("nhwc", [False, True])
@pytest.mark.parametrize(
    "weight_sizes,packed_shape,error",
    [
        # Two input channels per filter against a four-channel input: groups=2.
        pytest.param([4, 1, 1, 2], (4,), "must match input channels", id="grouped"),
        pytest.param([4, 1, 1, 4], (7,), "two per byte", id="wrong_byte_count"),
        pytest.param([4, 1, 1, 4], (2, 4), "must be 1-D", id="not_1d"),
        pytest.param([4, 0, 1, 4], (0,), "positive sizes", id="zero_size"),
    ],
)
def test_conv2d_s4_rejects_invalid_weights(
    fake, nhwc, weight_sizes, packed_shape, error
):
    op = (
        torch.ops.cortex_m.quantized_conv2d_s4_nhwc.default
        if nhwc
        else torch.ops.cortex_m.quantized_conv2d_s4.default
    )
    x = torch.ones((1, 4, 3, 3), dtype=torch.int8).to(memory_format=torch.channels_last)
    if nhwc:
        x = x.permute(0, 2, 3, 1).contiguous()
    args = (
        x,
        torch.zeros(packed_shape, dtype=torch.int8),
        None,
        (1, 1),
        (0, 0),
        (1, 1),
        0,
        0,
        torch.full((4,), 1 << 30, dtype=torch.int32),
        torch.ones(4, dtype=torch.int32),
        -128,
        127,
        weight_sizes,
        torch.empty(0, dtype=torch.uint8),
    )
    with FakeTensorMode(allow_non_fake_inputs=True) if fake else nullcontext():
        with pytest.raises(ValueError, match=error):
            op(*args)


a8w4_test_cases = {
    # 3 * 5 * 3 * 3 = 135 weights: odd, so the last high nibble is padding.
    "conv2d_odd_weight_count": McuTestCase(
        model=CortexMConv2DBias(3, 5, 3, padding=1),
        example_inputs=(
            ramp_tensor(-5, 5, (1, 3, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
    "conv2d_1x1": McuTestCase(
        model=CortexMConv2DBias(8, 4, 1),
        example_inputs=(
            ramp_tensor(-3, 3, (1, 8, 6, 6)).to(memory_format=torch.channels_last),
        ),
    ),
    # A single input channel would take the int8 depthwise kernel.
    "conv2d_single_input_channel": McuTestCase(
        model=CortexMDepthwiseConv2D(1, 4, 3, groups=1),
        example_inputs=(
            ramp_tensor(0, 10, (1, 1, 8, 8)).to(memory_format=torch.channels_last),
        ),
    ),
}

# Single-row convolutions, which CMSIS-NN routes to its 1xN kernel unless the
# s4 op sends them to arm_convolve_s4.
a8w4_conv1d_test_cases = {
    "conv1d_padding": McuTestCase(
        model=CortexMConv1D(4, 6, 3, padding=1),
        example_inputs=(ramp_tensor(-2, 2, (1, 4, 16)),),
    ),
    "conv1d_1xn": McuTestCase(
        model=CortexMConv1D(4, 8, 5),
        example_inputs=(ramp_tensor(-2, 2, (1, 4, 16)),),
    ),
    "conv1d_odd_input_channels": McuTestCase(
        model=CortexMConv1D(1, 8, 5, stride=4, padding=2),
        example_inputs=(ramp_tensor(-2, 2, (1, 1, 33)),),
    ),
    "conv1d_stride_padding": McuTestCase(
        model=CortexMConv1D(4, 8, 3, stride=2, padding=1),
        example_inputs=(ramp_tensor(-2, 2, (1, 4, 8)),),
    ),
    "conv1d_padding_exceeds_output": McuTestCase(
        model=CortexMConv1D(4, 8, 3, padding=1),
        example_inputs=(ramp_tensor(-2, 2, (1, 4, 1)),),
    ),
    "conv1d_stride_no_padding": McuTestCase(
        model=CortexMConv1D(4, 8, 3, stride=2),
        example_inputs=(ramp_tensor(-2, 2, (1, 4, 8)),),
    ),
}


@pytest.mark.parametrize(
    "c_in,kernel_w,input_w,stride,pad,input_h,expected",
    [
        pytest.param(1, 5, 33, 4, 2, 1, True, id="odd_input_channels"),
        pytest.param(2, 16, 33, 2, 0, 1, True, id="input_channels_not_multiple_of_4"),
        pytest.param(4, 3, 8, 2, 1, 1, True, id="uneven_padding"),
        pytest.param(4, 3, 1, 1, 1, 1, True, id="padding_exceeds_output"),
        # total_pad == -1: C and Python remainders differ here.
        pytest.param(4, 3, 8, 2, 0, 1, False, id="stride_no_padding"),
        pytest.param(8, 5, 32, 1, 2, 1, False, id="symmetric_padding"),
        pytest.param(1, 5, 33, 4, 2, 8, False, id="not_single_row"),
    ],
)
def test_use_generic_s4_kernel(c_in, kernel_w, input_w, stride, pad, input_h, expected):
    output_w = (input_w + 2 * pad - kernel_w) // stride + 1
    assert (
        _use_generic_s4_kernel(
            [1, input_h, input_w, c_in],
            [8, 1, kernel_w, c_in],
            [1, input_h, output_w, 8],
            [0, pad],
            [1, stride],
            [1, 1],
        )
        == expected
    )


def _lower_a8w4(
    test_case, target_config, use_explicit_layout, config=A8W4_PER_CHANNEL_CONFIG
):
    quantizer = CortexMQuantizer(use_explicit_layout=use_explicit_layout)
    quantizer.set_node_target(torch.ops.aten.conv2d.default, config)
    # Only the explicit-layout quantizer has conv1d patterns.
    if use_explicit_layout:
        quantizer.set_node_target(torch.ops.aten.conv1d.default, config)
    tester = CortexMTester(
        test_case.model,
        test_case.get_example_inputs(use_explicit_layout),
        target_config=target_config,
    )
    tester.quantize(Quantize(quantizer))
    tester.export().to_edge()
    tester.run_passes(
        CortexMRunPasses(
            target_config=target_config, use_explicit_layout=use_explicit_layout
        )
    )
    return tester


@parametrize("test_case", a8w4_test_cases)
@pytest.mark.parametrize("use_explicit_layout", [False, True])
def test_dialect_conv2d_a8w4(test_case, use_explicit_layout, cortex_m_target):
    tester = _lower_a8w4(test_case, cortex_m_target, use_explicit_layout)
    tester.run_method_and_compare_outputs(inputs=tester.example_inputs, qtol=1)

    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    s4_conv = (
        exir_ops.edge.cortex_m.quantized_conv2d_s4_nhwc.default
        if use_explicit_layout
        else exir_ops.edge.cortex_m.quantized_conv2d_s4.default
    )
    [conv] = [n for n in program.graph.nodes if n.target == s4_conv]
    weight = get_param_tensor(program, conv.args[1])
    weight_sizes = conv.args[12]
    conv_weight = test_case.model.conv.weight
    assert weight_sizes == [conv_weight.shape[i] for i in (0, 2, 3, 1)]
    assert weight.shape == ((conv_weight.numel() + 1) // 2,)
    assert unpack_int4(weight, weight_sizes).abs().max() <= 7


@parametrize("test_case", a8w4_conv1d_test_cases)
def test_dialect_conv1d_a8w4(test_case, cortex_m_target):
    tester = _lower_a8w4(test_case, cortex_m_target, use_explicit_layout=True)
    tester.check_count(
        {
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_conv2d_s4_nhwc_default": 1
        }
    )
    tester.run_method_and_compare_outputs(inputs=tester.example_inputs, qtol=1)


@pytest.mark.parametrize(
    "c_in,kernel,input_hw,cpu,packed",
    [
        # Rows of 45 and 38 values: unaligned VLD2 on MVE.
        pytest.param(5, 3, (5, 5), CortexM.M55, False, id="generic_m55"),
        pytest.param(5, 3, (5, 5), CortexM.M4, True, id="generic_m4"),
        pytest.param(38, 1, (1, 2), CortexM.M55, False, id="1x1_m55"),
        # Rows shorter than 32 values, or a multiple of 4 values, are safe.
        pytest.param(3, 1, (3, 3), CortexM.M55, True, id="short_rows_m55"),
        pytest.param(40, 1, (1, 2), CortexM.M55, True, id="aligned_1x1_m55"),
        pytest.param(2, 4, (5, 5), CortexM.M55, True, id="aligned_generic_m55"),
    ],
)
def test_dialect_conv2d_a8w4_unaligned_rows_use_int8_kernels_on_mve(
    c_in, kernel, input_hw, cpu, packed
):
    test_case = McuTestCase(
        model=CortexMConv2DBias(c_in, 8, kernel),
        example_inputs=(
            ramp_tensor(-2, 2, (1, c_in, *input_hw)).to(
                memory_format=torch.channels_last
            ),
        ),
    )
    tester = _lower_a8w4(
        test_case, CortexMTargetConfig(cpu=cpu), use_explicit_layout=False
    )
    tester.run_method_and_compare_outputs(inputs=tester.example_inputs, qtol=1)
    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    [conv] = [n for n in program.graph.nodes if "conv2d" in str(n.target)]
    expected = (
        exir_ops.edge.cortex_m.quantized_conv2d_s4.default
        if packed
        else exir_ops.edge.cortex_m.quantized_conv2d.default
    )
    assert conv.target == expected
    if not packed:
        assert get_param_tensor(program, conv.args[1]).abs().max() <= 7


@parametrize(
    "test_case",
    {name: test_cases[name] for name in ("conv2d_groups", "depthwise_conv2d")},
)
def test_dialect_a8w4_keeps_int8_weights_for_grouped_conv2d(test_case, cortex_m_target):
    tester = _lower_a8w4(test_case, cortex_m_target, use_explicit_layout=False)
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(inputs=tester.example_inputs, qtol=1)

    program = tester.get_artifact(StageType.RUN_PASSES).exported_program()
    [conv] = [n for n in program.graph.nodes if "conv2d" in str(n.target)]
    assert get_param_tensor(program, conv.args[1]).abs().max() > 7


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


@parametrize(
    "test_case",
    {
        name: test_cases[name]
        for name in ("conv2d_groups_bias", "depthwise_conv2d_bias")
    },
)
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            get_symmetric_a8w4_quantization_config(is_qat=False), id="arm_a8w4"
        ),
        pytest.param(_per_tensor_a8w4_config(), id="per_tensor_a8w4"),
    ],
)
def test_dialect_grouped_conv2d_int4_range_weights_use_int8_kernels(
    test_case, config, cortex_m_target
):
    tester = _lower_a8w4(
        test_case, cortex_m_target, use_explicit_layout=False, config=config
    )
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(inputs=tester.example_inputs, qtol=1)


@parametrize("test_case", a8w4_test_cases)
@pytest.mark.parametrize("use_explicit_layout", [False, True])
def test_implementation_conv2d_a8w4(test_case, use_explicit_layout, cortex_m_target):
    tester = _lower_a8w4(test_case, cortex_m_target, use_explicit_layout)
    tester.to_executorch().serialize()
    tester.run_method_and_compare_outputs(inputs=tester.example_inputs, qtol=2)


@parametrize("test_case", a8w4_conv1d_test_cases)
def test_implementation_conv1d_a8w4(test_case, cortex_m_target):
    tester = _lower_a8w4(test_case, cortex_m_target, use_explicit_layout=True)
    tester.to_executorch().serialize()
    tester.run_method_and_compare_outputs(inputs=tester.example_inputs, qtol=2)
