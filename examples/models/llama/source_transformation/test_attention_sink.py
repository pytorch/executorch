# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.examples.models.llama.llama_transformer import ModelArgs

from executorch.examples.models.llama.source_transformation.attention_sink import (
    RopeWithAttentionSink,
)
from parameterized import parameterized


class RopeWithAttentionSinkTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.params = ModelArgs(use_kv_cache=True, enable_dynamic_shape=True)
        self.rope_with_attention_sink = RopeWithAttentionSink(params=self.params)

    @parameterized.expand(
        [
            [128, 127],  # Rotate left
            [128, 128],  # No rotation
            [128, 129],  # Rotate right
        ]
    )
    def test_rotate(self, original_position, new_position):
        seq_len = 32

        q = torch.rand(
            1, seq_len, self.params.n_heads, self.params.head_dim, dtype=torch.float32
        )
        k = torch.rand(
            1,
            seq_len,
            self.params.n_heads,
            self.params.head_dim,
            dtype=torch.float32,
        )
        freqs_cos, freqs_sin = self.rope_with_attention_sink.get_freqs(
            input_pos=torch.tensor([original_position], dtype=torch.int32),
            seq_len=seq_len,
        )
        _, pre_rotated_k = self.rope_with_attention_sink.forward(
            q=q,
            k=k,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
        )

        rerotated_k = self.rope_with_attention_sink.rerotate_k(
            k=pre_rotated_k,
            original_position=original_position,
            new_position=new_position,
        )

        freqs_cos, freqs_sin = self.rope_with_attention_sink.get_freqs(
            input_pos=torch.tensor([new_position], dtype=torch.int32),
            seq_len=seq_len,
        )
        _, expected_k = self.rope_with_attention_sink.forward(
            q=q,
            k=k,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
        )

        torch.testing.assert_close(rerotated_k, expected_k)
