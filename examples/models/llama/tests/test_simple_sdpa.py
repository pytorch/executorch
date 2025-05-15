# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.examples.models.llama.attention import KVCache, SDPA
from executorch.examples.models.llama.source_transformation.sdpa import SDPASimple


class SDPATest(unittest.TestCase):
    def test_simple_sdpa(self):
        # Verify the correctness between the simple SDPA and the original SDPA module defined in llama_transformer.py
        max_batch_size = 1
        max_context_length = 128
        n_heads = 8
        head_dim = 8
        dim = 64
        n_rep = 1
        bsz = 1
        seqlen = 1
        n_local_heads = n_heads
        kv_cache = KVCache(
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            n_heads=n_heads,
            head_dim=head_dim,
            enable_dynamic_shape=False,
        )
        sdpa = SDPA(
            dim=dim,
            head_dim=head_dim,
            n_rep=n_rep,
            max_context_len=max_context_length,
        )
        input_pos = torch.tensor([0])
        query = torch.randn(1, 1, n_local_heads, head_dim)
        key = torch.randn(1, 1, n_local_heads, head_dim)
        value = torch.randn(1, 1, n_local_heads, head_dim)
        mask = torch.randn(max_context_length, max_context_length)
        mask = mask[input_pos]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        key, value = kv_cache.update(input_pos, key, value)

        sdpa_output = sdpa(
            input_pos,
            query,
            key,
            value,
            bsz=bsz,
            seqlen=seqlen,
            mask=mask,
        )

        simple_sdpa = SDPASimple(dim=dim, head_dim=head_dim, n_rep=n_rep)
        simple_sdpa_output = simple_sdpa(
            input_pos, query, key, value, bsz=bsz, seqlen=seqlen, mask=mask
        )

        # Compare the output from output from two sdpa implementation
        self.assertTrue(torch.allclose(sdpa_output, simple_sdpa_output))
