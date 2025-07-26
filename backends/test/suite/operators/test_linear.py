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
        in_features=10,
        out_features=5,
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
            ((torch.rand(2, 10) * 10).to(dtype),),
            flow,
        )

    @dtype_test
    def test_linear_no_bias_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            Model(bias=False).to(dtype),
            ((torch.rand(2, 10) * 10).to(dtype),),
            flow,
        )

    def test_linear_basic(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (torch.randn(2, 10),),
            flow,
        )

    def test_linear_feature_sizes(self, flow: TestFlow) -> None:
        self._test_op(
            Model(in_features=5, out_features=3),
            (torch.randn(2, 5),),
            flow,
        )
        self._test_op(
            Model(in_features=20, out_features=10),
            (torch.randn(2, 20),),
            flow,
        )
        self._test_op(
            Model(in_features=100, out_features=1),
            (torch.randn(2, 100),),
            flow,
        )
        self._test_op(
            Model(in_features=1, out_features=100),
            (torch.randn(2, 1),),
            flow,
        )

    def test_linear_no_bias(self, flow: TestFlow) -> None:
        self._test_op(
            Model(bias=False),
            (torch.randn(2, 10),),
            flow,
        )
        self._test_op(
            Model(in_features=20, out_features=15, bias=False),
            (torch.randn(2, 20),),
            flow,
        )

    def test_linear_batch_sizes(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (torch.randn(1, 10),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(5, 10),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(100, 10),),
            flow,
        )

    def test_linear_unbatched(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (torch.randn(10),),
            flow,
        )

    def test_linear_multi_dim_input(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (torch.randn(3, 4, 10),),
            flow,
        )
        self._test_op(
            Model(),
            (torch.randn(2, 3, 4, 10),),
            flow,
        )
