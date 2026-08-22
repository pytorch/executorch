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
        self.maxpool = torch.nn.MaxPool3d(
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
class MaxPool3d(OperatorTest):
    @dtype_test
    def test_maxpool3d_dtype(self, flow: TestFlow, dtype) -> None:
        # Input shape: (batch_size, channels, depth, height, width)
        self._test_op(
            Model().to(dtype),
            ((torch.rand(1, 4, 8, 8, 8) * 10).to(dtype),),
            flow,
        )

    def test_maxpool3d_kernel_size(self, flow: TestFlow) -> None:
        # Test with different kernel sizes
        self._test_op(
            Model(kernel_size=1),
            (torch.randn(1, 4, 8, 8, 8),),
            flow,
        )
        self._test_op(
            Model(kernel_size=(1, 2, 2)),
            (torch.randn(1, 4, 8, 8, 8),),
            flow,
        )

    def test_maxpool3d_stride(self, flow: TestFlow) -> None:
        # Test with different stride values
        self._test_op(
            Model(stride=2),
            (torch.randn(1, 4, 8, 8, 8),),
            flow,
        )
        self._test_op(
            Model(stride=(1, 2, 2)),
            (torch.randn(1, 4, 8, 8, 8),),
            flow,
        )

    def test_maxpool3d_padding(self, flow: TestFlow) -> None:
        # Test with different padding values
        self._test_op(
            Model(padding=1),
            (torch.randn(1, 4, 8, 8, 8),),
            flow,
        )
        self._test_op(
            Model(padding=(0, 1, 1)),
            (torch.randn(1, 4, 8, 8, 8),),
            flow,
        )

    def test_maxpool3d_dilation(self, flow: TestFlow) -> None:
        # Test with different dilation values
        self._test_op(
            Model(dilation=2),
            (torch.randn(1, 4, 8, 8, 8),),
            flow,
        )
        self._test_op(
            Model(dilation=(1, 2, 2)),
            (torch.randn(1, 4, 8, 8, 8),),
            flow,
        )

    def test_maxpool3d_ceil_mode(self, flow: TestFlow) -> None:
        # Test with ceil_mode=True
        self._test_op(
            Model(ceil_mode=True),
            (torch.randn(1, 4, 8, 8, 8),),
            flow,
        )

    def test_maxpool3d_return_indices(self, flow: TestFlow) -> None:
        # Test with return_indices=True
        class ModelWithIndices(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.maxpool = torch.nn.MaxPool3d(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    return_indices=True,
                )

            def forward(self, x):
                # Return both output and indices
                return self.maxpool(x)

        # Create a test input tensor
        input_tensor = torch.randn(1, 4, 8, 8, 8)

        self._test_op(
            Model(kernel_size=3, stride=2, padding=1),
            (input_tensor,),
            flow,
        )

    def test_maxpool3d_batch_sizes(self, flow: TestFlow) -> None:
        # Test with batch inputs
        self._test_op(
            Model(),
            (torch.randn(2, 4, 8, 8, 8),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(8, 4, 8, 8, 8),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(16, 4, 8, 8, 8),),
            flow,
        )

    def test_maxpool3d_input_sizes(self, flow: TestFlow) -> None:
        # Test with different input sizes
        self._test_op(
            Model(),
            (torch.randn(1, 2, 8, 8, 8),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(1, 8, 8, 8, 8),),
            flow,
        )

    def test_maxpool3d_combinations(self, flow: TestFlow) -> None:
        # Test with combinations of parameters
        self._test_op(
            Model(kernel_size=2, stride=2, padding=1),
            (torch.randn(1, 4, 8, 8, 8),),
            flow,
        )
        self._test_op(
            Model(kernel_size=3, stride=2, padding=1, ceil_mode=True),
            (torch.randn(1, 4, 10, 10, 10),),
            flow,
        )
        self._test_op(
            Model(
                kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1), dilation=2
            ),
            (torch.randn(1, 4, 8, 10, 10),),
            flow,
        )
