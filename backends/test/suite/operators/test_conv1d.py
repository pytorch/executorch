# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import torch
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)


class Model(torch.nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=6,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        return self.conv(x)


@operator_test
class Conv1d(OperatorTest):
    @dtype_test
    def test_conv1d_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            Model().to(dtype),
            ((torch.rand(4, 3, 50) * 10).to(dtype),),
            flow,
        )

    def test_conv1d_basic(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (torch.randn(4, 3, 50),),
            flow,
        )

    def test_conv1d_kernel_size(self, flow: TestFlow) -> None:
        self._test_op(
            Model(kernel_size=1),
            (torch.randn(4, 3, 50),),
            flow,
        )
        self._test_op(
            Model(kernel_size=5),
            (torch.randn(4, 3, 50),),
            flow,
        )

    def test_conv1d_stride(self, flow: TestFlow) -> None:
        self._test_op(
            Model(stride=2),
            (torch.randn(4, 3, 50),),
            flow,
        )

    def test_conv1d_padding(self, flow: TestFlow) -> None:
        self._test_op(
            Model(padding=1),
            (torch.randn(4, 3, 50),),
            flow,
        )
        self._test_op(
            Model(padding=2),
            (torch.randn(4, 3, 50),),
            flow,
        )

    def test_conv1d_dilation(self, flow: TestFlow) -> None:
        self._test_op(
            Model(dilation=2),
            (torch.randn(4, 3, 50),),
            flow,
        )

    def test_conv1d_groups(self, flow: TestFlow) -> None:
        self._test_op(
            Model(in_channels=6, out_channels=6, groups=3),
            (torch.randn(4, 6, 50),),
            flow,
        )

    def test_conv1d_depthwise(self, flow: TestFlow) -> None:
        self._test_op(
            Model(in_channels=8, out_channels=8, groups=8),
            (torch.randn(4, 8, 50),),
            flow,
        )

    def test_conv1d_no_bias(self, flow: TestFlow) -> None:
        self._test_op(
            Model(bias=False),
            (torch.randn(4, 3, 50),),
            flow,
        )

    def test_conv1d_padding_modes(self, flow: TestFlow) -> None:
        for mode in ["zeros", "reflect", "replicate", "circular"]:
            self._test_op(
                Model(padding=1, padding_mode=mode),
                (torch.randn(4, 3, 50),),
                flow,
            )

    def test_conv1d_channels(self, flow: TestFlow) -> None:
        self._test_op(
            Model(in_channels=1, out_channels=1),
            (torch.randn(4, 1, 50),),
            flow,
        )
        self._test_op(
            Model(in_channels=5, out_channels=10),
            (torch.randn(4, 5, 50),),
            flow,
        )
