# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List, Optional, Tuple, Union

import torch
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)


class MeanModel(torch.nn.Module):
    def __init__(
        self,
        dim: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
        keepdim: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim, dtype=self.dtype)


@operator_test
class Mean(OperatorTest):
    @dtype_test
    def test_mean_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            MeanModel().to(dtype),
            (torch.rand(10, 10).to(dtype),),
            flow,
        )

    def test_mean_basic(self, flow: TestFlow) -> None:
        self._test_op(
            MeanModel(),
            (torch.randn(10, 10),),
            flow,
        )

    def test_mean_dim(self, flow: TestFlow) -> None:
        self._test_op(
            MeanModel(dim=0),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            MeanModel(dim=1),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            MeanModel(dim=0),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(dim=1),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(dim=2),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(dim=1),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(dim=-1),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(dim=-2),
            (torch.randn(3, 4, 5),),
            flow,
        )

    def test_mean_multi_dim(self, flow: TestFlow) -> None:
        self._test_op(
            MeanModel(dim=(0, 1)),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(dim=(0, 2)),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(dim=(1, 2)),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(dim=(1, 3)),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(dim=(0, 2)),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(dim=(-1, -3)),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(dim=(0, 1, 2, 3)),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

    def test_mean_keepdim(self, flow: TestFlow) -> None:
        self._test_op(
            MeanModel(dim=0, keepdim=True),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            MeanModel(dim=1, keepdim=True),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            MeanModel(dim=1, keepdim=True),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(dim=2, keepdim=True),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(dim=(1, 2), keepdim=True),
            (torch.randn(3, 4, 5),),
            flow,
        )

    def test_mean_output_dtype(self, flow: TestFlow) -> None:
        self._test_op(
            MeanModel(dtype=torch.float32),
            (torch.randint(0, 10, (5, 10)),),
            flow,
        )

        self._test_op(
            MeanModel(dtype=torch.float64),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            MeanModel(dim=1, dtype=torch.float64),
            (torch.randn(5, 10),),
            flow,
        )

    def test_mean_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            MeanModel(),
            (torch.randn(20),),
            flow,
        )
        self._test_op(
            MeanModel(dim=0),
            (torch.randn(20),),
            flow,
        )

        self._test_op(
            MeanModel(),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            MeanModel(),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

        self._test_op(
            MeanModel(),
            (torch.randn(2, 2, 3, 4, 5),),
            flow,
        )

    def test_mean_edge_cases(self, flow: TestFlow) -> None:
        x = torch.tensor([[1.0, float("inf"), 3.0], [4.0, 5.0, float("inf")]])
        self._test_op(
            MeanModel(),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            MeanModel(dim=0),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            MeanModel(dim=1),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )

        x = torch.tensor([[1.0, float("-inf"), 3.0], [4.0, 5.0, float("-inf")]])
        self._test_op(
            MeanModel(),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            MeanModel(dim=0),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            MeanModel(dim=1),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )

        x = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
        self._test_op(
            MeanModel(),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            MeanModel(dim=0),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            MeanModel(dim=1),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )

    def test_mean_scalar(self, flow: TestFlow) -> None:
        self._test_op(
            MeanModel(),
            (torch.tensor([5.0]),),
            flow,
        )
        self._test_op(
            MeanModel(dim=0),
            (torch.tensor([5.0]),),
            flow,
        )
