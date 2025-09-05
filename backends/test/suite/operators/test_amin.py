# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List, Optional, Tuple, Union

import torch
import unittest
from executorch.backends.test.suite.flow import TestFlow

from executorch.backends.test.suite.operators import (
    dtype_test,
    operator_test,
    OperatorTest,
)


class AminModel(torch.nn.Module):
    def __init__(
        self,
        dim: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
        keepdim: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        if self.dim is None:
            return torch.amin(x, keepdim=self.keepdim)
        return torch.amin(x, dim=self.dim, keepdim=self.keepdim)


@operator_test
class Amin(OperatorTest):
    @dtype_test
    def test_amin_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            AminModel().to(dtype),
            (torch.rand(10, 10).to(dtype),),
            flow,
        )

    def test_amin_dim(self, flow: TestFlow) -> None:
        self._test_op(
            AminModel(dim=0),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            AminModel(dim=1),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            AminModel(dim=0),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(dim=1),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(dim=2),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(dim=1),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(dim=-1),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(dim=-2),
            (torch.randn(3, 4, 5),),
            flow,
        )

    def test_amin_multi_dim(self, flow: TestFlow) -> None:
        self._test_op(
            AminModel(dim=(0, 1)),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(dim=(0, 2)),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(dim=(1, 2)),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(dim=(1, 3)),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(dim=(0, 2)),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(dim=(-1, -3)),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(dim=(0, 1, 2, 3)),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

    def test_amin_keepdim(self, flow: TestFlow) -> None:
        self._test_op(
            AminModel(dim=0, keepdim=True),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            AminModel(dim=1, keepdim=True),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            AminModel(dim=1, keepdim=True),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(dim=2, keepdim=True),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(dim=(1, 2), keepdim=True),
            (torch.randn(3, 4, 5),),
            flow,
        )

    def test_amin_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            AminModel(),
            (torch.randn(20),),
            flow,
        )
        self._test_op(
            AminModel(dim=0),
            (torch.randn(20),),
            flow,
        )

        self._test_op(
            AminModel(),
            (torch.randn(5, 10),),
            flow,
        )

        self._test_op(
            AminModel(),
            (torch.randn(3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(),
            (torch.randn(2, 3, 4, 5),),
            flow,
        )

        self._test_op(
            AminModel(),
            (torch.randn(2, 2, 3, 4, 5),),
            flow,
        )

    @unittest.skip("NaN and Inf are not enforced for backends.")
    def test_amin_edge_cases(self, flow: TestFlow) -> None:
        x = torch.tensor([[1.0, float("-inf"), 3.0], [4.0, 5.0, float("-inf")]])
        self._test_op(
            AminModel(),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            AminModel(dim=0),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            AminModel(dim=1),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )

        x = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
        self._test_op(
            AminModel(),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            AminModel(dim=0),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            AminModel(dim=1),
            (x,),
            flow,
            generate_random_test_inputs=False,
        )

    def test_amin_scalar(self, flow: TestFlow) -> None:
        self._test_op(
            AminModel(),
            (torch.tensor([5.0]),),
            flow,
        )
        self._test_op(
            AminModel(dim=0),
            (torch.tensor([5.0]),),
            flow,
        )
