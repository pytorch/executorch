# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for TurboQuantKVCache module.

Verifies KV cache update/roundtrip quality, nibble packing correctness,
codebook/rotation properties, and torch.export compatibility.

No CUDA required — all tests run on CPU.

Usage:
    python -m pytest extension/llm/modules/test/test_turboquant_kv_cache.py -v
"""

import unittest

import torch

from executorch.extension.llm.modules.turboquant import TurboQuantKVCache
from executorch.extension.llm.modules.turboquant.codebook import (
    generate_rotation_matrix,
    solve_lloyd_max,
)
from torch.export import Dim, export

HEAD_DIM = 128
N_HEADS = 2
MAX_SEQ_LEN = 32
BITS = 4


def _roundtrip_cosine(cache, x, input_pos):
    """Update cache and measure roundtrip quality via cosine similarity.

    Reconstructs from the returned packed/norms using the test's own
    decompress logic (independent of cache internals).
    """
    k_p, k_n, _, _ = cache.update(input_pos, x, x)
    T = x.shape[2]

    # Decompress using centroids + rotation from the cache (public buffers)
    flat = k_p[:, :, input_pos].reshape(-1, HEAD_DIM // 2)
    flat_norms = k_n[:, :, input_pos].reshape(-1, 1).float()
    high = (flat >> 4).long()
    low = (flat & 0x0F).long()
    indices = torch.stack([high, low], dim=-1).reshape(-1, HEAD_DIM)
    reconstructed = cache.centroids.float()[indices]
    unrotated = reconstructed @ cache.rotation_T.float().T
    recon = (unrotated * flat_norms).reshape(1, N_HEADS, T, HEAD_DIM)

    return torch.nn.functional.cosine_similarity(
        x.reshape(-1, HEAD_DIM).float(),
        recon.reshape(-1, HEAD_DIM).float(),
    ).mean()


class TestCacheUpdate(unittest.TestCase):
    """update() produces high-quality compressed representation."""

    def test_roundtrip_quality(self):
        cache = TurboQuantKVCache(N_HEADS, HEAD_DIM, MAX_SEQ_LEN, BITS)
        x = torch.randn(1, N_HEADS, 10, HEAD_DIM)
        pos = torch.arange(10)
        cos = _roundtrip_cosine(cache, x, pos)
        self.assertGreater(cos.item(), 0.99)

    def test_output_shapes(self):
        cache = TurboQuantKVCache(N_HEADS, HEAD_DIM, MAX_SEQ_LEN, BITS)
        x = torch.randn(1, N_HEADS, 5, HEAD_DIM)
        k_p, k_n, v_p, v_n = cache.update(torch.arange(5), x, x)

        self.assertEqual(k_p.shape, (1, N_HEADS, MAX_SEQ_LEN, HEAD_DIM // 2))
        self.assertEqual(k_p.dtype, torch.uint8)
        self.assertEqual(k_n.shape, (1, N_HEADS, MAX_SEQ_LEN, 1))
        self.assertEqual(v_p.shape, k_p.shape)
        self.assertEqual(v_n.shape, k_n.shape)

    def test_bf16_input(self):
        cache = TurboQuantKVCache(N_HEADS, HEAD_DIM, MAX_SEQ_LEN, BITS)
        x = torch.randn(1, N_HEADS, 5, HEAD_DIM, dtype=torch.bfloat16)
        pos = torch.arange(5)
        cos = _roundtrip_cosine(cache, x, pos)
        self.assertGreater(cos.item(), 0.99)

    def test_state_accumulates(self):
        """Writing to different positions preserves earlier data."""
        cache = TurboQuantKVCache(N_HEADS, HEAD_DIM, MAX_SEQ_LEN, BITS)

        x0 = torch.randn(1, N_HEADS, 4, HEAD_DIM)
        cache.update(torch.arange(4), x0, x0)

        x1 = torch.randn(1, N_HEADS, 4, HEAD_DIM)
        k_p, k_n, _, _ = cache.update(torch.arange(4, 8), x1, x1)

        # Positions 0-3 should still have x0's data
        cos = _roundtrip_cosine(cache, x0, torch.arange(4))
        self.assertGreater(cos.item(), 0.99)

        # Positions 8+ should be zero (never written)
        self.assertEqual(k_p[:, :, 8:].abs().max().item(), 0)

    def test_head_dim_256(self):
        """Qwen 3.5 MoE config."""
        cache = TurboQuantKVCache(2, 256, 64)
        x = torch.randn(1, 2, 5, 256)
        k_p, k_n, v_p, v_n = cache.update(torch.arange(5), x, x)
        self.assertEqual(k_p.shape, (1, 2, 64, 128))
        self.assertEqual(k_p.dtype, torch.uint8)


class TestNibblePacking(unittest.TestCase):
    """uint8 nibble pack/unpack is bit-exact."""

    def test_roundtrip_all_index_pairs(self):
        all_pairs = torch.stack(
            torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij"),
            dim=-1,
        ).reshape(-1, 2)

        packed = (all_pairs[:, 0].to(torch.uint8) << 4) | all_pairs[:, 1].to(
            torch.uint8
        )

        high = (packed >> 4).long()
        low = (packed & 0x0F).long()

        self.assertTrue(torch.equal(high, all_pairs[:, 0].long()))
        self.assertTrue(torch.equal(low, all_pairs[:, 1].long()))


class TestTorchExport(unittest.TestCase):
    """TurboQuantKVCache survives torch.export(strict=True)."""

    def test_export_standalone(self):
        cache = TurboQuantKVCache(N_HEADS, HEAD_DIM, MAX_SEQ_LEN, BITS)
        seq_dim = Dim("seq", min=1, max=MAX_SEQ_LEN - 1)

        with torch.no_grad():
            ep = export(
                cache,
                args=(
                    torch.arange(2),
                    torch.randn(1, N_HEADS, 2, HEAD_DIM),
                    torch.randn(1, N_HEADS, 2, HEAD_DIM),
                ),
                dynamic_shapes={
                    "input_pos": {0: seq_dim},
                    "k_val": {2: seq_dim},
                    "v_val": {2: seq_dim},
                },
                strict=True,
            )

        mod = ep.module()
        k = torch.randn(1, N_HEADS, 3, HEAD_DIM)
        v = torch.randn(1, N_HEADS, 3, HEAD_DIM)
        k_p, k_n, v_p, v_n = mod(torch.arange(3), k, v)

        self.assertEqual(k_p.shape, (1, N_HEADS, MAX_SEQ_LEN, HEAD_DIM // 2))
        self.assertEqual(k_n.shape, (1, N_HEADS, MAX_SEQ_LEN, 1))

    def test_exported_state_accumulates(self):
        cache = TurboQuantKVCache(N_HEADS, HEAD_DIM, MAX_SEQ_LEN, BITS)
        seq_dim = Dim("seq", min=1, max=MAX_SEQ_LEN - 1)

        with torch.no_grad():
            ep = export(
                cache,
                args=(
                    torch.arange(2),
                    torch.randn(1, N_HEADS, 2, HEAD_DIM),
                    torch.randn(1, N_HEADS, 2, HEAD_DIM),
                ),
                dynamic_shapes={
                    "input_pos": {0: seq_dim},
                    "k_val": {2: seq_dim},
                    "v_val": {2: seq_dim},
                },
                strict=True,
            )

        mod = ep.module()

        # Write positions 0-1
        k0 = torch.randn(1, N_HEADS, 2, HEAD_DIM)
        mod(torch.arange(2), k0, torch.randn(1, N_HEADS, 2, HEAD_DIM))

        # Write positions 2-3, get full cache back
        k_p, k_n, _, _ = mod(
            torch.arange(2, 4),
            torch.randn(1, N_HEADS, 2, HEAD_DIM),
            torch.randn(1, N_HEADS, 2, HEAD_DIM),
        )

        # Positions 0-1 should be non-zero (k0's data preserved)
        self.assertGreater(k_p[:, :, :2].abs().max().item(), 0)

        # Positions 4+ should be zero (never written)
        self.assertEqual(k_p[:, :, 4:].abs().max().item(), 0)


class TestCodebook(unittest.TestCase):
    """Lloyd-Max codebook and rotation matrix correctness."""

    def test_centroids_sorted(self):
        centroids, boundaries = solve_lloyd_max(HEAD_DIM, BITS)
        self.assertEqual(centroids.shape, (16,))
        self.assertEqual(boundaries.shape, (15,))
        self.assertTrue(torch.all(centroids[1:] > centroids[:-1]))
        self.assertTrue(torch.all(boundaries[1:] > boundaries[:-1]))

    def test_centroids_symmetric(self):
        """Codebook should be roughly symmetric around zero."""
        centroids, _ = solve_lloyd_max(HEAD_DIM, BITS)
        self.assertAlmostEqual(centroids.mean().item(), 0.0, places=4)

    def test_boundaries_between_centroids(self):
        centroids, boundaries = solve_lloyd_max(HEAD_DIM, BITS)
        for i in range(len(boundaries)):
            self.assertGreater(boundaries[i].item(), centroids[i].item())
            self.assertLess(boundaries[i].item(), centroids[i + 1].item())

    def test_codebook_deterministic(self):
        c1, b1 = solve_lloyd_max(HEAD_DIM, BITS)
        c2, b2 = solve_lloyd_max(HEAD_DIM, BITS)
        self.assertTrue(torch.equal(c1, c2))
        self.assertTrue(torch.equal(b1, b2))

    def test_codebook_varies_with_dim(self):
        c64, _ = solve_lloyd_max(64, BITS)
        c256, _ = solve_lloyd_max(256, BITS)
        self.assertFalse(torch.allclose(c64, c256))

    def test_rotation_orthogonal(self):
        R = generate_rotation_matrix(HEAD_DIM)
        self.assertEqual(R.shape, (HEAD_DIM, HEAD_DIM))
        eye = R @ R.T
        self.assertTrue(torch.allclose(eye, torch.eye(HEAD_DIM), atol=1e-5))

    def test_rotation_deterministic(self):
        R1 = generate_rotation_matrix(HEAD_DIM, seed=42)
        R2 = generate_rotation_matrix(HEAD_DIM, seed=42)
        self.assertTrue(torch.equal(R1, R2))

    def test_rotation_varies_with_seed(self):
        R1 = generate_rotation_matrix(HEAD_DIM, seed=42)
        R2 = generate_rotation_matrix(HEAD_DIM, seed=99)
        self.assertFalse(torch.equal(R1, R2))


class TestEdgeCases(unittest.TestCase):

    def test_odd_head_dim_raises(self):
        with self.assertRaises(ValueError):
            TurboQuantKVCache(2, 127, 32)


if __name__ == "__main__":
    unittest.main()
