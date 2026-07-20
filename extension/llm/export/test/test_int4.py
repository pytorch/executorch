# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ExportableInt4Tensor + the torchao::dequantize_int4_tensor op."""

import unittest

import torch
from executorch.extension.llm.export.int4 import ExportableInt4Tensor


def _make_int4_tensor(N: int, K: int, gs: int, seed: int = 0):
    """Build a synthetic ``Int4Tensor`` plus the (q, scale, zero_point) it encodes.

    Returns ``(int4_tensor, q_unsigned (N,K), scale (K//gs,N), zero (K//gs,N))``.
    """
    from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor

    g = torch.Generator().manual_seed(seed)
    q = torch.randint(0, 16, (N, K), generator=g, dtype=torch.int32)  # unsigned [0,15]
    # Pack two nibbles/byte: even index -> low, odd -> high.
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)
    scale = (torch.randn(K // gs, N, generator=g) * 0.1).to(torch.bfloat16)
    zero = torch.randint(0, 16, (K // gs, N), generator=g).to(torch.bfloat16)
    it = Int4Tensor(
        qdata=packed,
        scale=scale,
        zero_point=zero,
        block_size=[1, gs],
        shape=torch.Size([N, K]),
    )
    return it, q, scale, zero


def _reference_dequant(q, scale, zero, gs):
    """Independent affine dequant: scale * (q - zero), groups expanded."""
    s = scale.t().to(torch.float32).repeat_interleave(gs, dim=-1)
    z = zero.t().to(torch.float32).repeat_interleave(gs, dim=-1)
    return (q.to(torch.float32) - z) * s


class TestDequantizeInt4Op(unittest.TestCase):
    def test_op_matches_reference(self):
        it, q, scale, zero = _make_int4_tensor(N=8, K=64, gs=32)
        out = torch.ops.torchao.dequantize_int4_tensor(
            it.qdata, it.scale, it.zero_point, 32, torch.float32
        )
        ref = _reference_dequant(q, scale, zero, 32)
        self.assertEqual(tuple(out.shape), (8, 64))
        self.assertTrue(torch.allclose(out, ref, rtol=1e-2, atol=5e-2))

    def test_subclass_dequantize_matches_op(self):
        it, _, _, _ = _make_int4_tensor(N=8, K=64, gs=32)
        t = ExportableInt4Tensor.from_int4_tensor(it)
        ref = torch.ops.torchao.dequantize_int4_tensor(
            it.qdata, it.scale, it.zero_point, 32, torch.bfloat16
        )
        self.assertTrue(torch.equal(t.dequantize(torch.bfloat16), ref))

    def test_subclass_linear_dispatches_to_dequant(self):
        it, _, _, _ = _make_int4_tensor(N=16, K=64, gs=32)
        t = ExportableInt4Tensor.from_int4_tensor(it)
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        out = torch.nn.functional.linear(x, t)
        ref = torch.nn.functional.linear(x, t.dequantize(torch.bfloat16))
        self.assertTrue(torch.equal(out, ref))

    def test_subclass_embedding_dispatches_to_dequant(self):
        it, _, _, _ = _make_int4_tensor(N=16, K=64, gs=32)
        t = ExportableInt4Tensor.from_int4_tensor(it)
        idx = torch.tensor([0, 3, 7, 1])
        out = torch.nn.functional.embedding(idx, t)
        ref = torch.nn.functional.embedding(idx, t.dequantize(torch.bfloat16))
        self.assertTrue(torch.equal(out, ref))


class TestExportableInt4TensorExport(unittest.TestCase):
    """Exporting a module whose weight is an ``ExportableInt4Tensor`` should lower
    linear/embedding through ``torchao::dequantize_int4_tensor`` after
    ``run_decompositions`` (the subclass dispatch fires during decomposition)."""

    @staticmethod
    def _targets(ep):
        return {str(n.target) for n in ep.graph.nodes if n.op == "call_function"}

    def test_linear_exports_with_dequantize_int4(self):
        it, _, _, _ = _make_int4_tensor(N=16, K=64, gs=32)
        t = ExportableInt4Tensor.from_int4_tensor(it)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(t, requires_grad=False)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.w)

        ep = torch.export.export(
            M(), (torch.randn(2, 64, dtype=torch.bfloat16),)
        ).run_decompositions({})
        self.assertIn("torchao.dequantize_int4_tensor.default", self._targets(ep))

    def test_embedding_exports_with_dequantize_int4(self):
        it, _, _, _ = _make_int4_tensor(N=16, K=64, gs=32)
        t = ExportableInt4Tensor.from_int4_tensor(it)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(t, requires_grad=False)

            def forward(self, idx):
                return torch.nn.functional.embedding(idx, self.w)

        ep = torch.export.export(M(), (torch.tensor([0, 1, 2, 3]),)).run_decompositions(
            {}
        )
        self.assertIn("torchao.dequantize_int4_tensor.default", self._targets(ep))


if __name__ == "__main__":
    unittest.main()
