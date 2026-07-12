#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``extension/llm/export/gguf.py``.

The reference oracle is the ``gguf`` package's own ``gguf.dequantize`` (which can
dequantize Q4_K / Q5_K / Q6_K). We validate that:

* ``ExportableGGUFTensor.dequantize`` (and the ``torchao::dequantize_gguf`` op,
  whose eager body uses ``gguf``) reproduces ``gguf.dequantize``;
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
    _max_uniform_group_size,
    _Q4_K_BLOCK_BYTES,
    _Q5_K_BLOCK_BYTES,
    _Q6_K_BLOCK_BYTES,
    ExportableGGUFTensor,
    Q4_K_GROUP_SIZE,
    Q5_K_GROUP_SIZE,
    Q6_K_GROUP_SIZE,
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


def _make_q5k_raw(N: int, nb: int, seed: int = 0) -> torch.Tensor:
    """A ``(N, nb*176)`` uint8 Q5_K blob with sane, deterministic magnitudes."""
    g = torch.Generator().manual_seed(seed)
    blk = torch.randint(
        0, 256, (N * nb, _Q5_K_BLOCK_BYTES), dtype=torch.uint8, generator=g
    )
    blk[:, 0:2] = _fp16_bytes(0.01)  # d
    blk[:, 2:4] = _fp16_bytes(0.01)  # dmin
    blk[:, 4:16] = 0x21  # fixed mid-range 6-bit sub-scales/mins (non-zero)
    return blk.reshape(N, nb * _Q5_K_BLOCK_BYTES)


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
            ("q5_k", GGMLQuantizationType.Q5_K, _make_q5k_raw),
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
        # the Intx tensor's dequantize exercises our unpacking (Q4_K/Q5_K/Q6_K).
        for ggml_type, make in (
            ("q4_k", _make_q4k_raw),
            ("q5_k", _make_q5k_raw),
            ("q6_k", _make_q6k_raw),
        ):
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

    def test_group_upgrade_default_keeps_native(self):
        # Without max_group_size the native group size is preserved.
        for ggml_type, make, native_gs in (
            ("q4_k", _make_q4k_raw, Q4_K_GROUP_SIZE),
            ("q5_k", _make_q5k_raw, Q5_K_GROUP_SIZE),
            ("q6_k", _make_q6k_raw, Q6_K_GROUP_SIZE),
        ):
            t = ExportableGGUFTensor.from_raw(make(N=2, nb=2), ggml_type)
            ix = t.to_intx_unpacked_to_int8_tensor()
            self.assertEqual(ix.block_size[1], native_gs, ggml_type)

    def test_group_upgrade_noop_on_nonuniform(self):
        # Non-uniform sub-block scales must not merge even with max_group_size
        # set, so Q6_K stays at its native group size 16 (< 32) -- the signal
        # repack_mlx uses to return None and fall back to the fused kernels.
        raw = _make_q6k_raw(N=3, nb=2)
        blk = raw.reshape(3 * 2, _Q6_K_BLOCK_BYTES)
        # Distinct int8 sub-block scales so adjacent groups differ.
        blk[:, 192:208] = torch.arange(1, 17, dtype=torch.int8).view(torch.uint8)
        t = ExportableGGUFTensor.from_raw(raw, "q6_k")
        ix = t.to_intx_unpacked_to_int8_tensor(max_group_size=128)
        self.assertEqual(ix.block_size[1], Q6_K_GROUP_SIZE)

    def test_group_upgrade_is_lossless(self):
        # The crafted blobs use fixed sub-scales/mins that are uniform within
        # each 64-wide run, so max_group_size=64 merges to group_size 64. The
        # merge must be bit-identical to the native-group unpacking (only equal
        # groups are merged) and still match the gguf reference.
        for ggml_type, make, native_gs in (
            ("q4_k", _make_q4k_raw, Q4_K_GROUP_SIZE),
            ("q5_k", _make_q5k_raw, Q5_K_GROUP_SIZE),
            ("q6_k", _make_q6k_raw, Q6_K_GROUP_SIZE),
        ):
            raw = make(N=3, nb=2)
            t = ExportableGGUFTensor.from_raw(raw, ggml_type)
            base = t.to_intx_unpacked_to_int8_tensor()
            up = t.to_intx_unpacked_to_int8_tensor(max_group_size=64)
            self.assertEqual(base.block_size[1], native_gs, ggml_type)
            self.assertEqual(up.block_size[1], 64, ggml_type)
            # qdata is untouched by the merge; only scale/zero_point subsample.
            self.assertTrue(torch.equal(up.qdata, base.qdata), ggml_type)
            # Bit-identical to the native-group unpacking (lossless).
            self.assertTrue(torch.equal(up.dequantize(), base.dequantize()), ggml_type)
            # ...and still matches the gguf reference within bf16 tolerance.
            self.assertTrue(
                torch.allclose(
                    up.dequantize().float(),
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

    def test_dequantize_gguf_op_matches_reference(self):
        for ggml_type, make in (
            ("q4_k", _make_q4k_raw),
            ("q5_k", _make_q5k_raw),
            ("q6_k", _make_q6k_raw),
        ):
            raw = make(N=3, nb=2)
            t = ExportableGGUFTensor.from_raw(raw, ggml_type)
            out = torch.ops.torchao.dequantize_gguf(raw, ggml_type, torch.float32)
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
            ExportableGGUFTensor.from_raw(raw, "q2_k")


@unittest.skipUnless(_HAS_GGUF, "gguf package not installed")
class TestExportableGGUFTensorExport(unittest.TestCase):
    """Exporting a module whose weight is an ``ExportableGGUFTensor`` should
    lower linear/embedding through the ``torchao::dequantize_gguf`` op after
    ``run_decompositions`` (the subclass dispatch fires during decomposition)."""

    @staticmethod
    def _targets(ep):
        return {str(n.target) for n in ep.graph.nodes if n.op == "call_function"}

    def test_linear_exports_with_dequantize_gguf(self):
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
        self.assertIn("torchao.dequantize_gguf.default", self._targets(ep))

    def test_embedding_exports_with_dequantize_gguf(self):
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
        self.assertIn("torchao.dequantize_gguf.default", self._targets(ep))


class TestMaxUniformGroupSize(unittest.TestCase):
    """Unit tests for the lossless group-size upgrade search.

    Exercises ``_max_uniform_group_size`` directly on synthetic scale/min arrays
    (no gguf package required).
    """

    def test_all_equal_upgrades_to_cap(self):
        gs, s, m = _max_uniform_group_size(torch.ones(3, 8), torch.zeros(3, 8), 32, 128)
        self.assertEqual(gs, 128)
        self.assertEqual(tuple(s.shape), (3, 2))
        self.assertEqual(tuple(m.shape), (3, 2))

    def test_cap_respected(self):
        gs, _, _ = _max_uniform_group_size(torch.ones(3, 8), torch.zeros(3, 8), 32, 64)
        self.assertEqual(gs, 64)

    def test_pairs_equal_upgrades_to_64(self):
        s = torch.tensor([[1, 1, 2, 2, 3, 3, 4, 4]] * 3, dtype=torch.float32)
        gs, _, _ = _max_uniform_group_size(s, torch.zeros(3, 8), 32, 128)
        self.assertEqual(gs, 64)

    def test_min_mismatch_blocks_merge(self):
        # Scales pair up, but a differing min forbids the (lossy) merge.
        s = torch.tensor([[1, 1, 2, 2, 3, 3, 4, 4]] * 3, dtype=torch.float32)
        m = torch.tensor([[0, 9, 0, 0, 0, 0, 0, 0]] * 3, dtype=torch.float32)
        gs, _, _ = _max_uniform_group_size(s, m, 32, 128)
        self.assertEqual(gs, 32)

    def test_distinct_stays_base_and_returns_inputs(self):
        s = torch.arange(8, dtype=torch.float32).repeat(2, 1)
        m = torch.zeros(2, 8)
        gs, s2, m2 = _max_uniform_group_size(s, m, 32, 128)
        self.assertEqual(gs, 32)
        self.assertTrue(torch.equal(s2, s))
        self.assertTrue(torch.equal(m2, m))

    def test_base16_all_equal(self):
        # Q6_K-style base group size of 16 can climb 16 -> 128 (factor 8).
        gs, _, _ = _max_uniform_group_size(
            torch.ones(2, 16), torch.zeros(2, 16), 16, 128
        )
        self.assertEqual(gs, 128)

    def test_subsampled_values(self):
        s = torch.tensor([[5, 5, 7, 7]] * 2, dtype=torch.float32)
        m = torch.tensor([[1, 1, 2, 2]] * 2, dtype=torch.float32)
        gs, s2, m2 = _max_uniform_group_size(s, m, 32, 64)
        self.assertEqual(gs, 64)
        self.assertTrue(
            torch.equal(s2, torch.tensor([[5, 7]] * 2, dtype=torch.float32))
        )
        self.assertTrue(
            torch.equal(m2, torch.tensor([[1, 2]] * 2, dtype=torch.float32))
        )


if __name__ == "__main__":
    unittest.main()
