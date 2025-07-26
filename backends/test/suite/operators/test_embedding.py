# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional

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
        num_embeddings=10,
        embedding_dim=5,
        padding_idx: Optional[int] = None,
        norm_type: float = 2.0,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            norm_type=norm_type,
        )

    def forward(self, x):
        return self.embedding(x)


@operator_test
class Embedding(OperatorTest):
    @dtype_test
    def test_embedding_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            Model().to(dtype),
            (torch.randint(0, 10, (2, 8), dtype=torch.long),),
            flow,
            generate_random_test_inputs=False,
        )

    def test_embedding_basic(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (torch.randint(0, 10, (2, 8), dtype=torch.long),),
            flow,
            generate_random_test_inputs=False,
        )

    def test_embedding_sizes(self, flow: TestFlow) -> None:
        self._test_op(
            Model(num_embeddings=5, embedding_dim=3),
            (torch.randint(0, 5, (2, 8), dtype=torch.long),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            Model(num_embeddings=100, embedding_dim=10),
            (torch.randint(0, 100, (2, 8), dtype=torch.long),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            Model(num_embeddings=1000, embedding_dim=50),
            (torch.randint(0, 1000, (2, 4), dtype=torch.long),),
            flow,
            generate_random_test_inputs=False,
        )

    def test_embedding_padding_idx(self, flow: TestFlow) -> None:
        self._test_op(
            Model(padding_idx=0),
            (torch.randint(0, 10, (2, 8), dtype=torch.long),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            Model(padding_idx=5),
            (torch.randint(0, 10, (2, 8), dtype=torch.long),),
            flow,
            generate_random_test_inputs=False,
        )

    def test_embedding_input_shapes(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (torch.randint(0, 10, (5,), dtype=torch.long),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            Model(),
            (torch.randint(0, 10, (2, 8), dtype=torch.long),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            Model(),
            (torch.randint(0, 10, (2, 3, 4), dtype=torch.long),),
            flow,
            generate_random_test_inputs=False,
        )
