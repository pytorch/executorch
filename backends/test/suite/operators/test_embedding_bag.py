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
        num_embeddings=10,
        embedding_dim=5,
        mode="mean",
        include_last_offset: bool = False,
    ):
        super().__init__()
        self.embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode=mode,
            include_last_offset=include_last_offset,
        )

    def forward(self, x, offsets=None):
        return self.embedding_bag(x, offsets)

    # Note that generate_random_test_inputs is used to avoid the tester
    # generating random inputs that are out of range of the embedding size.
    # The tester's random input generation is not smart enough to know that
    # the index inputs must be within a certain range.


@parameterize_by_dtype
def test_embedding_bag_dtype(test_runner, dtype) -> None:
    indices = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
    offsets = torch.tensor([0, 4], dtype=torch.long)
    test_runner.lower_and_run_model(
        Model().to(dtype),
        (indices, offsets),
        generate_random_test_inputs=False,
    )


def test_embedding_bag_sizes(test_runner) -> None:
    indices = torch.tensor([1, 2, 3, 1], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)

    test_runner.lower_and_run_model(
        Model(num_embeddings=5, embedding_dim=3),
        (indices, offsets),
        generate_random_test_inputs=False,
    )

    indices = torch.tensor([5, 20, 10, 43, 7], dtype=torch.long)
    offsets = torch.tensor([0, 2, 4], dtype=torch.long)
    test_runner.lower_and_run_model(
        Model(num_embeddings=50, embedding_dim=10),
        (indices, offsets),
        generate_random_test_inputs=False,
    )

    indices = torch.tensor([100, 200, 300, 400], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    test_runner.lower_and_run_model(
        Model(num_embeddings=500, embedding_dim=20),
        (indices, offsets),
        generate_random_test_inputs=False,
    )


def test_embedding_bag_modes(test_runner) -> None:
    indices = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
    offsets = torch.tensor([0, 4], dtype=torch.long)

    test_runner.lower_and_run_model(
        Model(mode="sum"),
        (indices, offsets),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        Model(mode="mean"),
        (indices, offsets),
        generate_random_test_inputs=False,
    )
    test_runner.lower_and_run_model(
        Model(mode="max"),
        (indices, offsets),
        generate_random_test_inputs=False,
    )


def test_embedding_bag_include_last_offset(test_runner) -> None:
    indices = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
    offsets = torch.tensor([0, 4], dtype=torch.long)

    test_runner.lower_and_run_model(
        Model(include_last_offset=True),
        (indices, offsets),
        generate_random_test_inputs=False,
    )
