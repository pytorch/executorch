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
        mode="mean",
        padding_idx: Optional[int] = None,
        norm_type: float = 2.0,
        include_last_offset: bool = False,
    ):
        super().__init__()
        self.embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode=mode,
            padding_idx=padding_idx,
            norm_type=norm_type,
            include_last_offset=include_last_offset,
        )

    def forward(self, x, offsets=None):
        return self.embedding_bag(x, offsets)


@operator_test
class EmbeddingBag(OperatorTest):
    @dtype_test
    def test_embedding_bag_dtype(self, flow: TestFlow, dtype) -> None:
        indices = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
        offsets = torch.tensor([0, 4], dtype=torch.long)
        self._test_op(
            Model().to(dtype),
            (indices, offsets),
            flow,
            generate_random_test_inputs=False,
        )

    def test_embedding_bag_basic(self, flow: TestFlow) -> None:
        indices = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
        offsets = torch.tensor([0, 4], dtype=torch.long)
        self._test_op(
            Model(),
            (indices, offsets),
            flow,
            generate_random_test_inputs=False,
        )

    def test_embedding_bag_sizes(self, flow: TestFlow) -> None:
        indices = torch.tensor([1, 2, 3, 1], dtype=torch.long)
        offsets = torch.tensor([0, 2], dtype=torch.long)

        self._test_op(
            Model(num_embeddings=5, embedding_dim=3),
            (indices, offsets),
            flow,
            generate_random_test_inputs=False,
        )

        indices = torch.tensor([5, 20, 10, 43, 7], dtype=torch.long)
        offsets = torch.tensor([0, 2, 4], dtype=torch.long)
        self._test_op(
            Model(num_embeddings=50, embedding_dim=10),
            (indices, offsets),
            flow,
            generate_random_test_inputs=False,
        )

        indices = torch.tensor([100, 200, 300, 400], dtype=torch.long)
        offsets = torch.tensor([0, 2], dtype=torch.long)
        self._test_op(
            Model(num_embeddings=500, embedding_dim=20),
            (indices, offsets),
            flow,
            generate_random_test_inputs=False,
        )

    def test_embedding_bag_modes(self, flow: TestFlow) -> None:
        indices = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
        offsets = torch.tensor([0, 4], dtype=torch.long)

        self._test_op(
            Model(mode="sum"),
            (indices, offsets),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            Model(mode="mean"),
            (indices, offsets),
            flow,
            generate_random_test_inputs=False,
        )
        self._test_op(
            Model(mode="max"),
            (indices, offsets),
            flow,
            generate_random_test_inputs=False,
        )

    def test_embedding_bag_padding_idx(self, flow: TestFlow) -> None:
        indices = torch.tensor([0, 1, 2, 0, 3, 0, 4], dtype=torch.long)
        offsets = torch.tensor([0, 3, 6], dtype=torch.long)

        self._test_op(
            Model(padding_idx=0),
            (indices, offsets),
            flow,
            generate_random_test_inputs=False,
        )

        indices = torch.tensor([1, 5, 2, 5, 3, 5, 4], dtype=torch.long)
        offsets = torch.tensor([0, 3, 6], dtype=torch.long)

        self._test_op(
            Model(padding_idx=5),
            (indices, offsets),
            flow,
            generate_random_test_inputs=False,
        )

    def test_embedding_bag_include_last_offset(self, flow: TestFlow) -> None:
        indices = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
        offsets = torch.tensor([0, 4], dtype=torch.long)

        self._test_op(
            Model(include_last_offset=True),
            (indices, offsets),
            flow,
            generate_random_test_inputs=False,
        )
