#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``extension/llm/export/gguf.py``.

The reference oracle is the ``gguf`` package's own ``gguf.dequantize`` (which can
dequantize Q4_K / Q6_K). We validate that:

* ``ExportableGGUFTensor.dequantize`` (and the fused ``torchao::gguf_*`` ops,
  whose eager bodies use ``gguf``) reproduce ``gguf.dequantize``;
* our hand-written ``to_int4_tensor`` / ``to_intx_unpacked_to_int8_tensor``
  unpack matches ``gguf.dequantize`` (within bf16 storage tolerance);
* using the subclass as a weight dispatches linear/embedding to the fused ops.

Blocks are crafted with a small fp16 super-block scale and fixed mid-range
sub-scales so dequantized magnitudes are O(1) and bf16 round-trip error is small
and deterministic (random sub-scales can produce near-zero effective scales,
which blow up the bf16 zero-point error for Q4_K).
"""

import unittest

import numpy as np
import torch

try:
    import gguf
    from gguf import GGMLQuantizationType

    _HAS_GGUF = True
except ImportError:
    _HAS_GGUF = False

from executorch.extension.llm.export.gguf import (
    _Q4_K_BLOCK_BYTES,
    _Q6_K_BLOCK_BYTES,
    ExportableGGUFTensor,
    Q4_K_GROUP_SIZE,
)


def _fp16_bytes(x: float) -> torch.Tensor:
    return torch.tensor([x], dtype=torch.float16).view(torch.uint8)


def _make_q4k_raw(N: int, nb: int, seed: int = 0) -> torch.Tensor:
    """A ``(N, nb*144)`` uint8 Q4_K blob with sane, deterministic magnitudes."""
    g = torch.Generator().manual_seed(seed)
    blk = torch.randint(
        0, 256, (N * nb, _Q4_K_BLOCK_BYTES), dtype=torch.uint8, generator=g
    )
    blk[:, 0:2] = _fp16_bytes(0.01)  # d
    blk[:, 2:4] = _fp16_bytes(0.01)  # dmin
    blk[:, 4:16] = 0x21  # fixed mid-range 6-bit sub-scales/mins (non-zero)
    return blk.reshape(N, nb * _Q4_K_BLOCK_BYTES)


def _make_q6k_raw(N: int, nb: int, seed: int = 0) -> torch.Tensor:
    """A ``(N, nb*210)`` uint8 Q6_K blob with sane, deterministic magnitudes."""
    g = torch.Generator().manual_seed(seed)
    blk = torch.randint(
        0, 256, (N * nb, _Q6_K_BLOCK_BYTES), dtype=torch.uint8, generator=g
    )
    blk[:, 192:208] = 0x10  # fixed int8 sub-scales (non-zero)
    blk[:, 208:210] = _fp16_bytes(0.01)  # d
    return blk.reshape(N, nb * _Q6_K_BLOCK_BYTES)


def _gguf_ref(raw: torch.Tensor, qtype) -> torch.Tensor:
    return torch.from_numpy(np.asarray(gguf.dequantize(raw.numpy(), qtype))).float()


def _int4_to_float(w) -> torch.Tensor:
    """Dequantize an ``Int4Tensor`` from its stored fields.

    ``Int4Tensor`` has no working ``dequantize()`` on CPU (``aten.dequantize`` is
    unimplemented and the linear path needs fbgemm), so reconstruct directly
    from its public fields (this still exercises our nibble-packing).
    """
    N, K = int(w.shape[0]), int(w.shape[1])
    gs = w.block_size[1]
    q = torch.empty(N, K, dtype=torch.float32)
    q[:, ::2] = (w.qdata & 0x0F).float()
    q[:, 1::2] = (w.qdata >> 4).float()
    scale = w.scale.t().float().repeat_interleave(gs, dim=1)
    zero = w.zero_point.t().float().repeat_interleave(gs, dim=1)
    return scale * (q - zero)


@unittest.skipUnless(_HAS_GGUF, "gguf package not installed")
class TestExportableGGUFTensor(unittest.TestCase):
    def test_dequantize_matches_gguf(self):
        for ggml_type, qtype, make in (
            ("q4_k", GGMLQuantizationType.Q4_K, _make_q4k_raw),
            ("q6_k", GGMLQuantizationType.Q6_K, _make_q6k_raw),
        ):
            raw = make(N=3, nb=2)
            t = ExportableGGUFTensor.from_raw(raw, ggml_type)
            self.assertEqual(tuple(t.shape), (3, 2 * 256))
            mine = t.dequantize(torch.float32)
            ref = _gguf_ref(raw, qtype)
            # .dequantize() routes through gguf, so it should match exactly.
            self.assertTrue(torch.equal(mine, ref), f"{qtype}")

    def test_to_intx_unpacked_matches_reference(self):
        # Reference is the gguf-package dequant (ExportableGGUFTensor.dequantize);
        # the Intx tensor's dequantize exercises our unpacking. Covers Q4_K & Q6_K.
        for ggml_type, make in (("q4_k", _make_q4k_raw), ("q6_k", _make_q6k_raw)):
            raw = make(N=3, nb=2)
            t = ExportableGGUFTensor.from_raw(raw, ggml_type)
            ix = t.to_intx_unpacked_to_int8_tensor()
            self.assertEqual(tuple(ix.shape), (3, 512))
            # bf16 storage tolerance.
            self.assertTrue(
                torch.allclose(
                    ix.dequantize().float(),
                    t.dequantize(torch.float32),
                    rtol=1e-2,
                    atol=5e-2,
                ),
                ggml_type,
            )

    def test_to_int4_tensor_matches_reference(self):
        raw = _make_q4k_raw(N=3, nb=2)
        t = ExportableGGUFTensor.from_raw(raw, "q4_k")
        w = t.to_int4_tensor()
        self.assertEqual(tuple(w.shape), (3, 512))
        self.assertEqual(list(w.block_size), [1, Q4_K_GROUP_SIZE])
        # Int4Tensor has no CPU dequantize(); reconstruct from its packed fields
        # (this still exercises our nibble-packing) against the gguf reference.
        self.assertTrue(
            torch.allclose(
                _int4_to_float(w),
                t.dequantize(torch.float32),
                rtol=1e-2,
                atol=5e-2,
            )
        )

    def test_gguf_dequantize_op_matches_reference(self):
        for ggml_type, make in (("q4_k", _make_q4k_raw), ("q6_k", _make_q6k_raw)):
            raw = make(N=3, nb=2)
            t = ExportableGGUFTensor.from_raw(raw, ggml_type)
            out = torch.ops.torchao.gguf_dequantize(raw, ggml_type, torch.float32)
            self.assertTrue(torch.equal(out, t.dequantize(torch.float32)))

    def test_subclass_linear_dispatches_to_dequant(self):
        raw = _make_q6k_raw(N=4, nb=1)
        t = ExportableGGUFTensor.from_raw(raw, "q6_k")
        x = torch.randn(2, 256, dtype=torch.bfloat16)
        out = torch.nn.functional.linear(x, t)
        ref = torch.nn.functional.linear(x, t.dequantize(torch.bfloat16))
        self.assertTrue(torch.equal(out, ref))

    def test_subclass_embedding_dispatches_to_dequant(self):
        raw = _make_q6k_raw(N=8, nb=1)
        t = ExportableGGUFTensor.from_raw(raw, "q6_k")
        idx = torch.tensor([0, 3, 7, 1])
        out = torch.nn.functional.embedding(idx, t)
        ref = torch.nn.functional.embedding(idx, t.dequantize(torch.bfloat16))
        self.assertTrue(torch.equal(out, ref))

    def test_unsupported_type_raises(self):
        raw = torch.zeros(1, _Q6_K_BLOCK_BYTES, dtype=torch.uint8)
        with self.assertRaises(NotImplementedError):
            ExportableGGUFTensor.from_raw(raw, "q5_k")


@unittest.skipUnless(_HAS_GGUF, "gguf package not installed")
class TestExportableGGUFTensorExport(unittest.TestCase):
    """Exporting a module whose weight is an ``ExportableGGUFTensor`` should
    lower linear/embedding through the ``torchao::gguf_dequantize`` op after
    ``run_decompositions`` (the subclass dispatch fires during decomposition)."""

    @staticmethod
    def _targets(ep):
        return {str(n.target) for n in ep.graph.nodes if n.op == "call_function"}

    def test_linear_exports_with_gguf_dequantize(self):
        t = ExportableGGUFTensor.from_raw(_make_q6k_raw(N=4, nb=1), "q6_k")

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(t, requires_grad=False)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.w)

        ep = torch.export.export(
            M(), (torch.randn(2, 256, dtype=torch.bfloat16),)
        ).run_decompositions({})
        self.assertIn("torchao.gguf_dequantize.default", self._targets(ep))

    def test_embedding_exports_with_gguf_dequantize(self):
        t = ExportableGGUFTensor.from_raw(_make_q6k_raw(N=8, nb=1), "q6_k")

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(t, requires_grad=False)

            def forward(self, idx):
                return torch.nn.functional.embedding(idx, self.w)

        ep = torch.export.export(M(), (torch.tensor([0, 1, 2, 3]),)).run_decompositions(
            {}
        )
        self.assertIn("torchao.gguf_dequantize.default", self._targets(ep))


if __name__ == "__main__":
    unittest.main()
