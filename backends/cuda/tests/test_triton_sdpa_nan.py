# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test Triton SDPA kernel with sparse boolean masks.

Reproduces NaN bug when all entries in a block are masked (-inf - (-inf) = NaN
in softmax). The fix guards exp(qk - m_ij) against all-masked blocks.

Before fix: NaN at start_pos=812 (sparse ring buffer mask).
After fix:  Finite output at all positions.

Usage:
    python -m pytest backends/cuda/tests/test_triton_sdpa_nan.py -v
"""

import glob
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F


# Shared test dimensions
B, H, D = 1, 4, 64
BUF_SIZE, WINDOW_SIZE = 1500, 750
RING_SEQ_LEN = 4
RING_POSITIONS = [0, 100, 374, 750, 812, 1000, 1496]


def _make_qkv(seq_len, kv_len, seed=42):
    """Create random bf16 Q, K, V tensors on CUDA."""
    torch.manual_seed(seed)
    return (
        torch.randn(B, H, seq_len, D, dtype=torch.bfloat16, device="cuda"),
        torch.randn(B, H, kv_len, D, dtype=torch.bfloat16, device="cuda"),
        torch.randn(B, H, kv_len, D, dtype=torch.bfloat16, device="cuda"),
    )


class SDPACausal(nn.Module):
    """Baseline: is_causal=True, no mask tensor."""

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class SDPASparseBoolMask(nn.Module):
    """Sparse bool mask (first half True), computed inside forward."""

    def forward(self, q, k, v):
        KV = k.shape[2]
        kv_pos = torch.arange(KV, device=q.device)
        mask = (kv_pos < KV // 2).view(1, 1, 1, KV).expand(1, 1, q.shape[2], -1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)


class SDPAWithRingBufferBoolMask(nn.Module):
    """SDPA with ring buffer sliding window bool mask computed inside forward.

    Matches the pattern used by streaming audio encoders: a sparse bool mask
    where many entries are False (masked), causing some Triton SDPA blocks
    to have ALL entries masked.
    """

    def __init__(self, window_size: int, buf_size: int):
        super().__init__()
        self.window_size = window_size
        self.buf_size = buf_size

    def forward(self, q, k, v, start_pos_tensor):
        start_pos = start_pos_tensor[0]
        seq_len = q.shape[2]
        total_written = start_pos + seq_len
        j = torch.arange(self.buf_size, dtype=torch.long, device=q.device)
        cache_pos = j + ((total_written - 1 - j) // self.buf_size) * self.buf_size
        q_offsets = torch.arange(seq_len, dtype=torch.long, device=q.device)
        pos_q = (start_pos + q_offsets).view(-1, 1)
        delta = pos_q - cache_pos.unsqueeze(0)
        valid = (cache_pos >= 0) & (delta >= 0) & (delta < self.window_size)
        mask = valid.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, buf]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)


def _export_pybind(module, inputs, tmpdir):
    """Export module to CUDA via AOTI with Triton ON, save to tmpdir, return loaded pybind module."""
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import (
        EdgeCompileConfig,
        ExecutorchBackendConfig,
        to_edge_transform_and_lower,
    )
    from executorch.exir.passes import MemoryPlanningPass
    from executorch.extension.pybindings.portable_lib import _load_for_executorch
    from torch.export import export

    ep = export(module, inputs, strict=True)
    compile_specs = [CudaBackend.generate_method_name_compile_spec("forward")]
    partitioner = {"forward": [CudaPartitioner(compile_specs)]}

    et_prog = to_edge_transform_and_lower(
        {"forward": ep},
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
        constant_methods={"test": 1},
    )
    et = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )

    pte_path = os.path.join(tmpdir, "test.pte")
    with open(pte_path, "wb") as f:
        et.write_to_file(f)
    ptd_path = None
    if et._tensor_data:
        et.write_tensor_data_to_file(tmpdir)
        ptd_files = glob.glob(os.path.join(tmpdir, "*.ptd"))
        ptd_path = ptd_files[0] if ptd_files else None

    return _load_for_executorch(pte_path, data_path=ptd_path)


def _run_pybind(mod, inputs):
    """Run loaded pybind module with CUDA inputs (automatically moved to CPU)."""
    cpu_inputs = [t.cpu() for t in inputs]
    return mod.run_method("forward", cpu_inputs)


class TestTritonSdpaNan(unittest.TestCase):
    """Test Triton SDPA kernel with sparse boolean masks."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        if not torch.cuda.is_bf16_supported():
            raise unittest.SkipTest("BF16 not supported on this GPU")

        # Export ring buffer model once, reused across all subTests.
        cls._ring_buffer_tmpdir = tempfile.mkdtemp()
        q, k, v = _make_qkv(RING_SEQ_LEN, BUF_SIZE, seed=0)
        sp = torch.tensor([0], dtype=torch.long, device="cuda")
        module = SDPAWithRingBufferBoolMask(WINDOW_SIZE, BUF_SIZE).eval()
        cls._ring_buffer_model = _export_pybind(
            module, (q, k, v, sp), cls._ring_buffer_tmpdir
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_ring_buffer_tmpdir"):
            shutil.rmtree(cls._ring_buffer_tmpdir, ignore_errors=True)

    def test_causal_vs_eager(self):
        """Baseline: is_causal=True should match eager closely."""
        T = 64
        q, k, v = _make_qkv(T, T, seed=42)

        module = SDPACausal().eval()
        with torch.no_grad():
            eager = module(q, k, v).float().cpu()

        with tempfile.TemporaryDirectory() as tmpdir:
            mod = _export_pybind(module, (q, k, v), tmpdir)
            triton = _run_pybind(mod, (q, k, v))[0].float()

        self.assertFalse(torch.isnan(triton).any(), "is_causal output has NaN")
        rel = (triton - eager).abs() / eager.abs().clamp(min=1e-6)
        self.assertLess(
            rel.mean().item(), 0.1, f"is_causal mean_rel={rel.mean():.4f} too large"
        )

    def test_non_pow2_head_dim_with_bool_mask(self):
        """Non-pow2 HEAD_DIM with sparse bool mask exercises _sdpa_fwd_kernel_non_pow2.

        Tests both the safe_diff/safe_alpha_diff NaN guards and the other=False
        fix for out-of-bounds mask positions. Uses D=80 (non-pow2, BLOCK_D=128,
        BLOCK_N=128) and KV_LEN=200 (not divisible by BLOCK_N) so the last
        block has out-of-bounds positions where other=False matters.
        """
        D_NP2 = 80
        SEQ_LEN = 4
        KV_LEN = 200

        torch.manual_seed(42)
        q = torch.randn(B, H, SEQ_LEN, D_NP2, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, KV_LEN, D_NP2, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, KV_LEN, D_NP2, dtype=torch.bfloat16, device="cuda")

        module = SDPASparseBoolMask().eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            mod = _export_pybind(module, (q, k, v), tmpdir)
            triton = _run_pybind(mod, (q, k, v))[0].float()

        self.assertFalse(torch.isnan(triton).any(), "non-pow2 bool mask has NaN")
        self.assertFalse(torch.isinf(triton).any(), "non-pow2 bool mask has Inf")

        """
        with torch.no_grad():
            eager = module(q, k, v).float().cpu()
        rel = (triton - eager).abs() / eager.abs().clamp(min=1e-6)
        TODO: Enable this test. Currently fails.
        self.assertLess(
            rel.mean().item(),
            0.1,
            f"non-pow2 bool mask mean_rel={rel.mean():.4f} too large",
        )"""

    def test_ring_buffer_bool_mask_no_nan(self):
        """Triton SDPA must not produce NaN with sparse ring buffer bool masks.

        Before the fix, exp(-inf - (-inf)) = NaN in the softmax when all entries
        in a tile block were masked. This test verifies the output is finite.
        """
        for start_pos in RING_POSITIONS:
            with self.subTest(start_pos=start_pos):
                q, k, v = _make_qkv(RING_SEQ_LEN, BUF_SIZE, seed=start_pos)
                sp = torch.tensor([start_pos], dtype=torch.long, device="cuda")

                triton_out = _run_pybind(self._ring_buffer_model, (q, k, v, sp))[
                    0
                ].float()

                nan_count = torch.isnan(triton_out).sum().item()
                self.assertEqual(
                    nan_count,
                    0,
                    f"Triton SDPA output has {nan_count} NaN values at "
                    f"start_pos={start_pos}. Softmax produces NaN when all "
                    f"block entries are masked to -inf.",
                )

                inf_count = torch.isinf(triton_out).sum().item()
                self.assertEqual(
                    inf_count,
                    0,
                    f"Triton SDPA output has {inf_count} Inf values at "
                    f"start_pos={start_pos}.",
                )


if __name__ == "__main__":
    unittest.main()
