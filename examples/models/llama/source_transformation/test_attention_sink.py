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

    def _init_rope(self, params: ModelArgs, eviction_batch_size: int):
        return RopeWithAttentionSink(
            params=params,
            window_size=252,
            sink_size=4,
            eviction_batch_size=eviction_batch_size,
        )

    def setUp(self):
        torch.manual_seed(42)
        self.params = ModelArgs(
            use_kv_cache=True, enable_dynamic_shape=True, max_seq_len=256
        )
        self.rope_with_attention_sink = self._init_rope(
            params=self.params, eviction_batch_size=1
        )

    @parameterized.expand(
        [
            [0, 10, 1, 0],  # No shift
            [250, 10, 1, 246],  # Some shift
            [256, 10, 1, 246],  # All shift
            [0, 10, 30, 0],  # No shift with batch eviction
            [250, 10, 30, 220],  # Some shift with batch eviction
            [256, 10, 30, 226],  # All shift with batch eviction
        ]
    )
    def test_get_freqs(
        self, input_pos, seq_len, eviction_batch_size, expected_result_pos
    ):
        self.rope_with_attention_sink = self._init_rope(
            params=self.params, eviction_batch_size=eviction_batch_size
        )

        freqs_cos, freqs_sin = self.rope_with_attention_sink.get_freqs(
            input_pos=torch.tensor([input_pos], dtype=torch.int32),
            seq_len=seq_len,
        )

        torch.testing.assert_close(
            freqs_cos,
            self.rope_with_attention_sink.freqs_cos.narrow(
                0, expected_result_pos, seq_len
            ),
        )
        torch.testing.assert_close(
            freqs_sin,
            self.rope_with_attention_sink.freqs_sin.narrow(
                0, expected_result_pos, seq_len
            ),
        )

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
