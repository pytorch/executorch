# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch

from executorch.backends.test.suite.operators import parameterize_by_dtype


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

    # Note that generate_random_test_inputs is used to avoid the tester
    # generating random inputs that are out of range of the embedding size.
    # The tester's random input generation is not smart enough to know that
    # the index inputs must be within a certain range.


@parameterize_by_dtype
def test_embedding_dtype(test_runner, dtype) -> None:
    test_runner.lower_and_run_model(
        Model().to(dtype),
        (torch.randint(0, 10, (2, 8), dtype=torch.long),),
        generate_random_test_inputs=False,
    )


def test_embedding_sizes(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(num_embeddings=5, embedding_dim=3),
        (torch.randint(0, 5, (2, 8), dtype=torch.long),),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        Model(num_embeddings=100, embedding_dim=10),
        (torch.randint(0, 100, (2, 8), dtype=torch.long),),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        Model(num_embeddings=1000, embedding_dim=50),
        (torch.randint(0, 1000, (2, 4), dtype=torch.long),),
        generate_random_test_inputs=False,
    )


def test_embedding_batch_dim(test_runner) -> None:
    test_runner.lower_and_run_model(
        Model(),
        (torch.randint(0, 100, (5,), dtype=torch.long),),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randint(0, 100, (2, 8), dtype=torch.long),),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        Model(),
        (torch.randint(0, 100, (2, 3, 4), dtype=torch.long),),
        generate_random_test_inputs=False,
    )
