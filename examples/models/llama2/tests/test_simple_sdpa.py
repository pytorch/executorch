# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from executorch.examples.models.llama2.llama_transformer import KVCache, SDPA
from executorch.examples.models.llama2.source_transformation.sdpa import SDPASimple


class SDPATest(unittest.TestCase):
    def test_simple_sdpa(self):
        # Verify the correctness between the simple SDPA and the original SDPA module defined in llama_transformer.py
        max_batch_size = 1
        max_seq_length = 128
        n_heads = 8
        head_dim = 8
        dim = 64
        n_rep = 1
        bsz = 1
        seqlen = 1
        n_local_heads = n_heads
        kv_cache = KVCache(
            max_batch_size=max_batch_size,
            max_seq_length=max_seq_length,
            n_heads=n_heads,
            head_dim=head_dim,
            transpose_cache=True,
            enable_dynamic_shape=False,
        )
        sdpa = SDPA(
            kv_cache=copy.deepcopy(kv_cache),
            dim=dim,
            head_dim=head_dim,
            n_rep=n_rep,
            max_seq_len=max_seq_length,
            enable_dynamic_shape=False,
        )
        input_pos = torch.tensor([0])
        query = torch.randn(1, 1, n_local_heads, head_dim)
        key = torch.randn(1, 1, n_local_heads, head_dim)
        value = torch.randn(1, 1, n_local_heads, head_dim)
        mask = torch.randn(max_seq_length, max_seq_length)
        sdpa_output = sdpa(
            input_pos,
            query,
            key,
            value,
            bsz=bsz,
            seqlen=seqlen,
            mask=mask,
        )

        simple_sdpa = SDPASimple(
            kv_cache=copy.deepcopy(kv_cache), dim=dim, head_dim=head_dim, n_rep=n_rep
        )
        simple_sdpa_output = simple_sdpa(
            input_pos, query, key, value, bsz=bsz, seqlen=seqlen, mask=mask
        )

        # Compare the output from output from two sdpa implementation
        self.assertTrue(torch.allclose(sdpa_output, simple_sdpa_output))
