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
        kernel_size=3,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def forward(self, x):
        return self.maxpool(x)


@operator_test
class MaxPool2d(OperatorTest):
    @dtype_test
    def test_maxpool2d_dtype(self, flow: TestFlow, dtype) -> None:
        # Input shape: (batch_size, channels, height, width)
        self._test_op(
            Model().to(dtype),
            ((torch.rand(1, 8, 20, 20) * 10).to(dtype),),
            flow,
        )

    def test_maxpool2d_kernel_size(self, flow: TestFlow) -> None:
        # Test with different kernel sizes
        self._test_op(
            Model(kernel_size=1),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )
        self._test_op(
            Model(kernel_size=5),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )
        self._test_op(
            Model(kernel_size=(3, 2)),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )

    def test_maxpool2d_stride(self, flow: TestFlow) -> None:
        # Test with different stride values
        self._test_op(
            Model(stride=2),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )
        self._test_op(
            Model(stride=(2, 1)),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )

    def test_maxpool2d_padding(self, flow: TestFlow) -> None:
        # Test with different padding values
        self._test_op(
            Model(padding=1),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )
        self._test_op(
            Model(padding=(1, 2)),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )

    def test_maxpool2d_dilation(self, flow: TestFlow) -> None:
        # Test with different dilation values
        self._test_op(
            Model(dilation=2),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )
        self._test_op(
            Model(dilation=(2, 1)),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )

    def test_maxpool2d_ceil_mode(self, flow: TestFlow) -> None:
        # Test with ceil_mode=True
        self._test_op(
            Model(ceil_mode=True),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )

    def test_maxpool2d_return_indices(self, flow: TestFlow) -> None:
        # Test with return_indices=True
        class ModelWithIndices(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.maxpool = torch.nn.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    return_indices=True,
                )

            def forward(self, x):
                return self.maxpool(x)

        # Create a test input tensor
        input_tensor = torch.randn(1, 8, 20, 20)

        self._test_op(
            Model(kernel_size=3, stride=2, padding=1),
            (input_tensor,),
            flow,
        )

    def test_maxpool2d_batch_sizes(self, flow: TestFlow) -> None:
        # Test with batch inputs
        self._test_op(
            Model(),
            (torch.randn(2, 8, 20, 20),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(8, 8, 20, 20),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(16, 8, 20, 20),),
            flow,
        )

    def test_maxpool2d_input_sizes(self, flow: TestFlow) -> None:
        # Test with different input sizes
        self._test_op(
            Model(),
            (torch.randn(1, 4, 20, 20),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(1, 16, 20, 20),),
            flow,
        )

    def test_maxpool2d_combinations(self, flow: TestFlow) -> None:
        # Test with combinations of parameters
        self._test_op(
            Model(kernel_size=2, stride=2, padding=1),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )
        self._test_op(
            Model(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            (torch.randn(1, 8, 21, 21),),
            flow,
        )
        self._test_op(
            Model(kernel_size=(2, 3), stride=(2, 1), padding=(1, 0), dilation=2),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )
