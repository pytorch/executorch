# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from functools import partial

import torch
from executorch.examples.models.llama.llama_transformer import ModelArgs
from executorch.examples.models.llama.rope import (
    hf_precompute_freqs_cis,
    precompute_freqs_cis,
)

from executorch.examples.models.llama.source_transformation.attention_sink import (
    RopeWithAttentionSink,
)


def _get_rope_with_attention_sink(params: ModelArgs) -> RopeWithAttentionSink:
    if params.use_hf_rope:
        precompute_freqs_cis_fn = hf_precompute_freqs_cis
    else:
        precompute_freqs_cis_fn = partial(
            precompute_freqs_cis, use_scaled=params.use_scaled_rope
        )
    freqs_cos, freqs_sin = precompute_freqs_cis_fn(
        params.dim // params.n_heads,
        (
            params.max_seq_len  # Normal llama2.
            if params.ffn_dim_multiplier is None
            else params.max_seq_len * 2  # Sharded checkpoint.
        ),
        params.rope_freq_base,
    )
    return RopeWithAttentionSink(
        params=params, freqs_cos=freqs_cos, freqs_sin=freqs_sin
    )


class RopeWithAttentionSinkTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.params = ModelArgs(use_kv_cache=True, enable_dynamic_shape=True)
        self.rope_with_attention_sink = _get_rope_with_attention_sink(self.params)
        self.seq_len = 32
        self.n_local_heads = self.params.n_heads
        self.n_local_kv_heads = self.params.n_heads
        self.head_dim = self.params.dim // self.params.n_heads
        self.q = torch.rand(
            1, self.seq_len, self.n_local_heads, self.head_dim, dtype=torch.float32
        )
        self.k = torch.rand(
            1,
            self.seq_len,
            self.n_local_kv_heads,
            self.head_dim,
            dtype=torch.float32,
        )

    def test_rotate_backward(self):
        original_position = 128
        new_position = 127

        _, pre_rotated_k = self.rope_with_attention_sink.forward(
            q=self.q,
            k=self.k,
            input_pos=torch.tensor([original_position], dtype=torch.int32),
        )

        k = self.rope_with_attention_sink.rerotate_k(
            k=pre_rotated_k,
            original_position=original_position,
            new_position=new_position,
        )

        _, expected_k = self.rope_with_attention_sink.forward(
            q=self.q,
            k=self.k,
            input_pos=torch.tensor([new_position], dtype=torch.int32),
        )

        torch.testing.assert_close(k, expected_k)

    def test_rotate_inplace(self):
        original_position = 128
        new_position = 128

        _, pre_rotated_k = self.rope_with_attention_sink.forward(
            q=self.q,
            k=self.k,
            input_pos=torch.tensor([original_position], dtype=torch.int32),
        )

        k = self.rope_with_attention_sink.rerotate_k(
            k=pre_rotated_k,
            original_position=original_position,
            new_position=new_position,
        )

        _, expected_k = self.rope_with_attention_sink.forward(
            q=self.q,
            k=self.k,
            input_pos=torch.tensor([new_position], dtype=torch.int32),
        )

        torch.testing.assert_close(k, expected_k)

    def test_rotate_forward(self):
        original_position = 128
        new_position = 129

        _, pre_rotated_k = self.rope_with_attention_sink.forward(
            q=self.q,
            k=self.k,
            input_pos=torch.tensor([original_position], dtype=torch.int32),
        )

        k = self.rope_with_attention_sink.rerotate_k(
            k=pre_rotated_k,
            original_position=original_position,
            new_position=new_position,
        )

        _, expected_k = self.rope_with_attention_sink.forward(
            q=self.q,
            k=self.k,
            input_pos=torch.tensor([new_position], dtype=torch.int32),
        )

        torch.testing.assert_close(k, expected_k)
