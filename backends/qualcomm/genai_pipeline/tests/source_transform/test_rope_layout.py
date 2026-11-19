# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the RoPE weight-layout state-dict transform."""

import unittest

import torch

from executorch.backends.qualcomm.genai_pipeline.source_transform.rope_layout import (
    permute_partial_rope,
)


class TestPermutePartialRope(unittest.TestCase):
    """Re-lays-out q/k weights from HF's interleaved RoPE to HTP's split form."""

    def test_permutes_query_and_key_weights(self):
        weight = torch.arange(8, dtype=torch.float32).reshape(8, 1)
        state_dict = {
            "layers.0.attention.wq.weight": weight.clone(),
            "layers.0.attention.wk.weight": weight.clone(),
        }

        result = permute_partial_rope(
            state_dict,
            partial_rotary_factor=1.0,
            n_layers=1,
            n_heads=2,
            n_kv_heads=2,
        )

        expected = torch.tensor(
            [[0.0], [2.0], [1.0], [3.0], [4.0], [6.0], [5.0], [7.0]]
        )
        torch.testing.assert_close(result["layers.0.attention.wq.weight"], expected)
        torch.testing.assert_close(result["layers.0.attention.wk.weight"], expected)

    def test_leaves_value_weights_untouched(self):
        """Only q and k feed RoPE; v must keep HF's layout."""
        weight = torch.arange(8, dtype=torch.float32).reshape(8, 1)
        state_dict = {
            "layers.0.attention.wq.weight": weight.clone(),
            "layers.0.attention.wk.weight": weight.clone(),
            "layers.0.attention.wv.weight": weight.clone(),
        }

        result = permute_partial_rope(
            state_dict,
            partial_rotary_factor=1.0,
            n_layers=1,
            n_heads=2,
            n_kv_heads=2,
        )

        torch.testing.assert_close(result["layers.0.attention.wv.weight"], weight)

    def test_uses_separate_head_counts_for_q_and_k(self):
        """GQA models have n_kv_heads < n_heads; each is permuted by its own count."""
        state_dict = {
            "layers.0.attention.wq.weight": torch.arange(
                8, dtype=torch.float32
            ).reshape(8, 1),
            "layers.0.attention.wk.weight": torch.arange(
                8, dtype=torch.float32
            ).reshape(8, 1),
        }

        result = permute_partial_rope(
            state_dict,
            partial_rotary_factor=1.0,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
        )

        self.assertFalse(
            torch.equal(
                result["layers.0.attention.wq.weight"],
                result["layers.0.attention.wk.weight"],
            )
        )

    def test_covers_every_layer(self):
        weight = torch.arange(8, dtype=torch.float32).reshape(8, 1)
        state_dict = {
            f"layers.{i}.attention.w{p}.weight": weight.clone()
            for i in range(3)
            for p in ("q", "k")
        }

        result = permute_partial_rope(
            state_dict,
            partial_rotary_factor=1.0,
            n_layers=3,
            n_heads=2,
            n_kv_heads=2,
        )

        for i in range(3):
            self.assertFalse(
                torch.equal(result[f"layers.{i}.attention.wq.weight"], weight)
            )


if __name__ == "__main__":
    unittest.main()
