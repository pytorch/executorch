# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext

import pytest
import torch
from executorch.backends.cortex_m.test.tester import CortexMTester
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("depthwise", [False, True])
@pytest.mark.parametrize("nhwc", [False, True])
@pytest.mark.parametrize(
    "size,kernel,stride,padding,error",
    [
        ((2, 2), (1, 1), (1, 1), (0, 0, 1, 1), "SAME convolution geometry"),
        ((1, 8), (1, 5), (1, 2), (0, 1, 0, 2), "unsupported for 1xN"),
    ],
)
def test_unsupported_four_value_padding(
    fake, depthwise, nhwc, size, kernel, stride, padding, error
):
    ops = {
        (False, False): torch.ops.cortex_m.quantized_conv2d.default,
        (False, True): torch.ops.cortex_m.quantized_conv2d_nhwc.default,
        (True, False): torch.ops.cortex_m.quantized_depthwise_conv2d.default,
        (True, True): torch.ops.cortex_m.quantized_depthwise_conv2d_nhwc.default,
    }
    x = torch.ones((1, 1, *size), dtype=torch.int8).to(
        memory_format=torch.channels_last
    )
    if nhwc:
        x = x.permute(0, 2, 3, 1).contiguous()
    weight = torch.ones((1, *kernel, 1), dtype=torch.int8)
    args = (x, weight, None, stride, padding, (1, 1))
    if depthwise:
        args += (1,)
    args += (
        0,
        0,
        torch.tensor([1 << 30], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        -128,
        127,
        torch.empty(0, dtype=torch.uint8),
    )
    with FakeTensorMode(allow_non_fake_inputs=True) if fake else nullcontext():
        with pytest.raises(ValueError, match=error):
            ops[depthwise, nhwc](*args)


class PaddedConv(torch.nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        groups,
        kernel,
        stride,
        padding,
        value=0,
        shared=False,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            channels, out_channels, kernel, stride=stride, groups=groups
        )
        self.padding = padding
        self.value = value
        self.shared = shared

    def forward(self, x):
        padded = torch.nn.functional.pad(x, self.padding, value=self.value)
        result = self.conv(padded)
        return (result, padded) if self.shared else result


pad_cases = [
    pytest.param(
        1, 64, 1, (10, 4), (2, 2), (1, 1, 4, 5), 0, False, True, id="ds_cnn_stem"
    ),
    pytest.param(
        1,
        8,
        1,
        (10, 4),
        (2, 2),
        (1, 1, 4, 5),
        0,
        False,
        True,
        id="single_channel_depthwise",
    ),
    pytest.param(4, 8, 1, (3, 3), (2, 2), (0, 1, 0, 1), 0, False, True, id="conv"),
    pytest.param(4, 8, 4, (3, 3), (2, 2), (0, 1, 0, 1), 0, False, True, id="depthwise"),
    pytest.param(4, 8, 2, (3, 3), (2, 2), (0, 1, 0, 1), 0, False, True, id="grouped"),
    pytest.param(4, 8, 1, (1, 5), (1, 2), (1, 2, 0, 0), 0, False, False, id="1xn"),
    pytest.param(
        4, 8, 1, (3, 3), (2, 2), (0, 1, 0, 1), 1.0, False, False, id="nonzero_pad"
    ),
    pytest.param(
        4, 8, 1, (3, 3), (2, 2), (1, 0, 1, 0), 0, False, False, id="same_lower"
    ),
    pytest.param(4, 8, 1, (3, 3), (2, 2), (0, 1, 0, 1), 0, True, True, id="shared_pad"),
]


@pytest.mark.parametrize(
    "channels,out_channels,groups,kernel,stride,padding,value,shared,fused", pad_cases
)
def test_fuse_conv_padding(
    channels, out_channels, groups, kernel, stride, padding, value, shared, fused
):
    model = PaddedConv(
        channels, out_channels, groups, kernel, stride, padding, value, shared
    ).eval()
    size = (49, 10) if channels == 1 else (8, 8)
    if kernel[0] == 1:
        size = (1, 8)
    x = (torch.rand(1, channels, *size) * 5 - 1).to(memory_format=torch.channels_last)
    tester = CortexMTester(model, (x,)).quantize().export().to_edge().run_passes()
    tester.run_method_and_compare_outputs(inputs=(x,), qtol=1)
    graph = tester.get_artifact(StageType.RUN_PASSES).exported_program().graph
    convs = [n for n in graph.nodes if "conv2d" in str(n.target)]
    assert len(convs) == 1
    conv = convs[0]
    assert ("depthwise" in str(conv.target)) == (
        groups == channels and not (channels == 1 and out_channels > 8)
    )
    assert (len(conv.args[4]) == 4) == fused
    pads = [n for n in graph.nodes if n.target == exir_ops.edge.cortex_m.pad.default]
    assert len(pads) == (not fused or shared)


@pytest.mark.parametrize(
    "channels,out_channels,groups,kernel,stride,padding,value,shared,fused",
    pad_cases[:6],
)
def test_implementation_fused_conv_padding(
    channels,
    out_channels,
    groups,
    kernel,
    stride,
    padding,
    value,
    shared,
    fused,
    cortex_m_target,
):
    model = PaddedConv(channels, out_channels, groups, kernel, stride, padding).eval()
    size = (49, 10) if channels == 1 else (8, 8)
    if kernel[0] == 1:
        size = (1, 8)
    x = (torch.rand(1, channels, *size) * 5 - 1).to(memory_format=torch.channels_last)
    CortexMTester(model, (x,), target_config=cortex_m_target).test_implementation(
        qtol=1
    )
