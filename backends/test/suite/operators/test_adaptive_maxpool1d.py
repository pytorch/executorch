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
        output_size=5,
        return_indices=False,
    ):
        super().__init__()
        self.adaptive_maxpool = torch.nn.AdaptiveMaxPool1d(
            output_size=output_size,
            return_indices=return_indices,
        )

    def forward(self, x):
        return self.adaptive_maxpool(x)


@operator_test
class AdaptiveMaxPool1d(OperatorTest):
    @dtype_test
    def test_adaptive_maxpool1d_dtype(self, flow: TestFlow, dtype) -> None:
        # Input shape: (batch_size, channels, length)
        self._test_op(
            Model().to(dtype),
            ((torch.rand(1, 8, 100) * 10).to(dtype),),
            flow,
        )

    def test_adaptive_maxpool1d_output_size(self, flow: TestFlow) -> None:
        # Test with different output sizes
        self._test_op(
            Model(output_size=1),
            (torch.randn(1, 8, 100),),
            flow,
        )
        self._test_op(
            Model(output_size=10),
            (torch.randn(1, 8, 100),),
            flow,
        )
        self._test_op(
            Model(output_size=50),
            (torch.randn(1, 8, 100),),
            flow,
        )

    def test_adaptive_maxpool1d_return_indices(self, flow: TestFlow) -> None:
        # Test with return_indices=True
        class ModelWithIndices(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adaptive_maxpool = torch.nn.AdaptiveMaxPool1d(
                    output_size=5,
                    return_indices=True,
                )

            def forward(self, x):
                return self.adaptive_maxpool(x)

        input_tensor = torch.randn(1, 8, 100)

        self._test_op(
            ModelWithIndices(),
            (input_tensor,),
            flow,
        )

    def test_adaptive_maxpool1d_batch_sizes(self, flow: TestFlow) -> None:
        # Test with batch inputs
        self._test_op(
            Model(),
            (torch.randn(2, 8, 100),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(8, 8, 100),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(16, 8, 100),),
            flow,
        )

    def test_adaptive_maxpool1d_input_sizes(self, flow: TestFlow) -> None:
        # Test with different input sizes
        self._test_op(
            Model(),
            (torch.randn(1, 4, 100),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(1, 16, 100),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(1, 8, 50),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(1, 8, 200),),
            flow,
        )
