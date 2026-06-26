# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester
from torch.export import Dim


def _hf_freqs(seq_len: int, head_dim: int, doubled: bool = True) -> tuple:
    """Generate cos/sin frequencies. If doubled, first and second halves are identical."""
    half = head_dim // 2
    freqs = torch.randn(seq_len, half)
    if doubled:
        emb = torch.cat((freqs, freqs), dim=-1)
    else:
        emb = torch.cat((freqs, torch.randn(seq_len, half)), dim=-1)
    return torch.cos(emb).unsqueeze(0), torch.sin(emb).unsqueeze(0)


class TestRope(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    class HFRope(torch.nn.Module):
        """HuggingFace-style rotary position embedding (split-half layout)."""

        def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            rot = torch.cat((-x2, x1), dim=-1)
            return (x * cos) + (rot * sin)

    def _test_rope(self, inputs, dynamic_shapes=None):
        (
            Tester(self.HFRope(), inputs, dynamic_shapes=dynamic_shapes)
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_fp32_rope(self):
        batch, seq_len, n_heads, head_dim = 1, 8, 4, 32
        cos, sin = _hf_freqs(seq_len, head_dim)
        inputs = (torch.randn(batch, seq_len, n_heads, head_dim), cos, sin)
        self._test_rope(inputs)

    def test_fp32_rope_large_head_dim(self):
        batch, seq_len, n_heads, head_dim = 1, 16, 8, 128
        cos, sin = _hf_freqs(seq_len, head_dim)
        inputs = (torch.randn(batch, seq_len, n_heads, head_dim), cos, sin)
        self._test_rope(inputs)

    def test_fp32_rope_dynamic_seq_len(self):
        batch, seq_len, n_heads, head_dim = 1, 8, 4, 32
        cos, sin = _hf_freqs(seq_len, head_dim)
        inputs = (torch.randn(batch, seq_len, n_heads, head_dim), cos, sin)
        seq = Dim("seq", min=1, max=128)
        dynamic_shapes = (
            {0: None, 1: seq, 2: None, 3: None},
            {0: None, 1: seq, 2: None},
            {0: None, 1: seq, 2: None},
        )
        self._test_rope(inputs, dynamic_shapes=dynamic_shapes)

    class HFRopeBHSD(torch.nn.Module):
        """HF-style RoPE with BHSD layout (transpose before/after RoPE)."""

        def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
            x = x.transpose(1, 2)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            rot = torch.cat((-x2, x1), dim=-1)
            out = (x * cos) + (rot * sin)
            return out.transpose(1, 2)

    def _test_rope_bhsd(self, inputs, dynamic_shapes=None):
        (
            Tester(self.HFRopeBHSD(), inputs, dynamic_shapes=dynamic_shapes)
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(inputs=inputs)
        )

    def test_fp32_rope_bhsd(self):
        batch, seq_len, n_heads, head_dim = 1, 8, 4, 32
        cos, sin = _hf_freqs(seq_len, head_dim)
        inputs = (torch.randn(batch, seq_len, n_heads, head_dim), cos, sin)
        self._test_rope_bhsd(inputs)

    def test_fp32_rope_bhsd_large_head_dim(self):
        batch, seq_len, n_heads, head_dim = 1, 16, 8, 128
        cos, sin = _hf_freqs(seq_len, head_dim)
        inputs = (torch.randn(batch, seq_len, n_heads, head_dim), cos, sin)
        self._test_rope_bhsd(inputs)

    def test_fp32_rope_bhsd_dynamic_seq_len(self):
        batch, seq_len, n_heads, head_dim = 1, 8, 4, 32
        cos, sin = _hf_freqs(seq_len, head_dim)
        inputs = (torch.randn(batch, seq_len, n_heads, head_dim), cos, sin)
        seq = Dim("seq", min=1, max=128)
        dynamic_shapes = (
            {0: None, 1: seq, 2: None, 3: None},
            {0: None, 1: seq, 2: None},
            {0: None, 1: seq, 2: None},
        )
        self._test_rope_bhsd(inputs, dynamic_shapes=dynamic_shapes)

    def test_non_doubled_freqs_not_fused(self):
        """Non-doubled cos/sin must not be fused into xnnpack.rope.

        The fused op only uses the first half of cos/sin as weights. If
        fusion fires on non-doubled frequencies (where the two halves
        differ), the second half is silently discarded and the output is
        wrong. This test catches that: run_method_and_compare_outputs will
        fail if fusion incorrectly applies.
        """
        batch, seq_len, n_heads, head_dim = 1, 8, 4, 32
        cos, sin = _hf_freqs(seq_len, head_dim, doubled=False)
        inputs = (torch.randn(batch, seq_len, n_heads, head_dim), cos, sin)
        (
            Tester(self.HFRope(), inputs)
            .export()
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs(inputs=inputs)
        )
