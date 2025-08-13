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
        output_size=(5, 5),
    ):
        super().__init__()
        self.adaptive_avgpool = torch.nn.AdaptiveAvgPool2d(
            output_size=output_size,
        )

    def forward(self, x):
        return self.adaptive_avgpool(x)


@operator_test
class AdaptiveAvgPool2d(OperatorTest):
    @dtype_test
    def test_adaptive_avgpool2d_dtype(self, flow: TestFlow, dtype) -> None:
        # Input shape: (batch_size, channels, height, width)
        self._test_op(
            Model().to(dtype),
            ((torch.rand(1, 8, 20, 20) * 10).to(dtype),),
            flow,
        )

    def test_adaptive_avgpool2d_output_size(self, flow: TestFlow) -> None:
        # Test with different output sizes
        self._test_op(
            Model(output_size=1),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )
        self._test_op(
            Model(output_size=(1, 1)),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )
        self._test_op(
            Model(output_size=(10, 10)),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )
        self._test_op(
            Model(output_size=(5, 10)),
            (torch.randn(1, 8, 20, 20),),
            flow,
        )

    def test_adaptive_avgpool2d_batch_sizes(self, flow: TestFlow) -> None:
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

    def test_adaptive_avgpool2d_input_sizes(self, flow: TestFlow) -> None:
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
        self._test_op(
            Model(),
            (torch.randn(1, 8, 10, 10),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(1, 8, 30, 30),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(1, 8, 15, 25),),
            flow,
        )
