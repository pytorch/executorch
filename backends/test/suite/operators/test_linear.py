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
        in_features=67,
        out_features=43,
        bias=True,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x):
        return self.linear(x)


@operator_test
class Linear(OperatorTest):
    @dtype_test
    def test_linear_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            Model().to(dtype),
            ((torch.rand(16, 64) * 10).to(dtype),),
            flow,
        )

    @dtype_test
    def test_linear_no_bias_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            Model(bias=False).to(dtype),
            ((torch.rand(16, 64) * 10).to(dtype),),
            flow,
        )

    def test_linear_feature_sizes(self, flow: TestFlow) -> None:
        self._test_op(
            Model(in_features=32, out_features=16),
            (torch.randn(20, 32),),
            flow,
        )
        self._test_op(
            Model(in_features=128, out_features=64),
            (torch.randn(8, 128),),
            flow,
        )
        self._test_op(
            Model(in_features=256, out_features=1),
            (torch.randn(4, 256),),
            flow,
        )
        self._test_op(
            Model(in_features=1, out_features=512),
            (torch.randn(1024, 1),),
            flow,
        )

    def test_linear_no_bias(self, flow: TestFlow) -> None:
        self._test_op(
            Model(bias=False),
            (torch.randn(16, 64),),
            flow,
        )
        self._test_op(
            Model(in_features=128, out_features=96, bias=False),
            (torch.randn(8, 128),),
            flow,
        )

    def test_linear_batch_sizes(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (torch.randn(8, 64),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(32, 64),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(100, 64),),
            flow,
        )

    def test_linear_unbatched(self, flow: TestFlow) -> None:
        self._test_op(
            Model(in_features=512),
            (torch.randn(512),),
            flow,
        )

    def test_linear_leading_batch(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (torch.randn(4, 8, 64),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(2, 4, 8, 64),),
            flow,
        )
