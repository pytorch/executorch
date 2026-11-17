# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the token-embedding state-dict transform."""

import unittest

import torch

from executorch.backends.qualcomm.genai_pipeline.source_transform.embedding_scale import (
    scale_token_embedding,
)


class TestScaleTokenEmbedding(unittest.TestCase):
    """Gemma-family models scale embeddings by ``sqrt(hidden_size)``."""

    def test_scales_by_factor(self):
        result = scale_token_embedding(
            {"tok_embeddings.weight": torch.ones(2)},
            embedding_scale_factor=2.0,
        )
        torch.testing.assert_close(
            result["tok_embeddings.weight"], torch.full((2,), 2.0)
        )

    def test_no_op_when_embedding_absent(self):
        """Models with a separated token-embedding graph carry no such key."""
        result = scale_token_embedding(
            {"norm.weight": torch.ones(2)},
            embedding_scale_factor=2.0,
        )
        torch.testing.assert_close(result["norm.weight"], torch.ones(2))

    def test_no_op_at_unit_factor(self):
        """Runs for every model, so a factor of 1.0 must change nothing at all --
        including dtype, which the multiply would otherwise upcast."""
        weight = torch.ones(2, dtype=torch.bfloat16)
        result = scale_token_embedding(
            {"tok_embeddings.weight": weight},
            embedding_scale_factor=1.0,
        )
        self.assertIs(result["tok_embeddings.weight"], weight)
        self.assertEqual(result["tok_embeddings.weight"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
