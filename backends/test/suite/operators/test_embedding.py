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
        num_embeddings=100,
        embedding_dim=50,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

    def forward(self, x):
        return self.embedding(x)


@operator_test
class Embedding(OperatorTest):
    # Note that generate_random_test_inputs is used to avoid the tester
    # generating random inputs that are out of range of the embedding size.
    # The tester's random input generation is not smart enough to know that
    # the index inputs must be within a certain range.

    @dtype_test
    def test_embedding_dtype(self, flow: TestFlow, dtype) -> None:
        self._test_op(
            Model().to(dtype),
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

    def test_embedding_batch_dim(self, flow: TestFlow) -> None:
        self._test_op(
            Model(),
            (torch.randint(0, 100, (5,), dtype=torch.long),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            Model(),
            (torch.randint(0, 100, (2, 8), dtype=torch.long),),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            Model(),
            (torch.randint(0, 100, (2, 3, 4), dtype=torch.long),),
            flow,
            generate_random_test_inputs=False,
        )
