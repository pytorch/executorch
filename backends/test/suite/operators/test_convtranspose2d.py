# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Tuple, Union

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
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv_transpose(x)


@operator_test
class ConvTranspose2d(OperatorTest):
    @dtype_test
    def test_convtranspose2d_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            Model().to(dtype),
            ((torch.rand(4, 3, 16, 16) * 10).to(dtype),),
            flow,
        )

    def test_convtranspose2d_basic(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (torch.randn(4, 3, 16, 16),),
            flow,
        )

    def test_convtranspose2d_kernel_size(self, flow: TestFlow) -> None:
        self._test_op(
            Model(kernel_size=1),
            (torch.randn(4, 3, 16, 16),),
            flow,
        )
        self._test_op(
            Model(kernel_size=5),
            (torch.randn(4, 3, 16, 16),),
            flow,
        )
        self._test_op(
            Model(kernel_size=(3, 5)),
            (torch.randn(4, 3, 16, 16),),
            flow,
        )

    def test_convtranspose2d_stride(self, flow: TestFlow) -> None:
        self._test_op(
            Model(stride=2),
            (torch.randn(4, 3, 16, 16),),
            flow,
        )
        self._test_op(
            Model(stride=(2, 1)),
            (torch.randn(4, 3, 16, 16),),
            flow,
        )

    def test_convtranspose2d_padding(self, flow: TestFlow) -> None:
        self._test_op(
            Model(padding=1),
            (torch.randn(4, 3, 16, 16),),
            flow,
        )
        self._test_op(
            Model(padding=(1, 2)),
            (torch.randn(4, 3, 16, 16),),
            flow,
        )

    def test_convtranspose2d_output_padding(self, flow: TestFlow) -> None:
        self._test_op(
            Model(stride=2, output_padding=1),
            (torch.randn(4, 3, 16, 16),),
            flow,
        )
        self._test_op(
            Model(stride=(2, 2), output_padding=(1, 0)),
            (torch.randn(4, 3, 16, 16),),
            flow,
        )

    def test_convtranspose2d_dilation(self, flow: TestFlow) -> None:
        self._test_op(
            Model(dilation=2),
            (torch.randn(4, 3, 16, 16),),
            flow,
        )
        self._test_op(
            Model(dilation=(2, 1)),
            (torch.randn(4, 3, 16, 16),),
            flow,
        )

    def test_convtranspose2d_groups(self, flow: TestFlow) -> None:
        self._test_op(
            Model(in_channels=6, out_channels=6, groups=3),
            (torch.randn(4, 6, 16, 16),),
            flow,
        )

    def test_convtranspose2d_depthwise(self, flow: TestFlow) -> None:
        self._test_op(
            Model(in_channels=8, out_channels=8, groups=8),
            (torch.randn(4, 8, 16, 16),),
            flow,
        )

    def test_convtranspose2d_no_bias(self, flow: TestFlow) -> None:
        self._test_op(
            Model(bias=False),
            (torch.randn(4, 3, 16, 16),),
            flow,
        )

    def test_convtranspose2d_channels(self, flow: TestFlow) -> None:
        self._test_op(
            Model(in_channels=1, out_channels=1),
            (torch.randn(4, 1, 16, 16),),
            flow,
        )
        self._test_op(
            Model(in_channels=5, out_channels=10),
            (torch.randn(4, 5, 16, 16),),
            flow,
        )

    def test_convtranspose2d_different_spatial_dims(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (torch.randn(4, 3, 20, 16),),
            flow,
        )
