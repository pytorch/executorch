# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test ReplaceEdgeOpWithTritonOpPass split-K SDPA kernel selection.

Exports a minimal model containing F.scaled_dot_product_attention through
the CUDA backend and verifies that the pass routes to split-K for decode
(L_q=1, large L_kv) and standard SDPA otherwise.
"""

import logging
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_cuda(tc: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        tc.skipTest("CUDA required")


class SDPAModule(nn.Module):
    """Single-layer model with SDPA and a static KV cache buffer."""

    def __init__(self, n_heads, n_kv_heads, head_dim, kv_len):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        hidden = n_heads * head_dim
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.register_buffer(
            "k_cache", torch.zeros(1, n_kv_heads, kv_len, head_dim), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(1, n_kv_heads, kv_len, head_dim), persistent=False
        )

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        self.k_cache.index_copy_(2, input_pos, k)
        self.v_cache.index_copy_(2, input_pos, v)
        y = F.scaled_dot_product_attention(
            q,
            self.k_cache,
            self.v_cache,
            enable_gqa=True,
        )
        return y.transpose(1, 2).contiguous().view(B, T, -1)


def _export_through_cuda_backend(model, example_args):
    """Export and lower through the CUDA backend (stops before to_executorch)."""
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torch.export import export

    with torch.no_grad():
        ep = export(model, example_args, strict=True)

    return to_edge_transform_and_lower(
        {"decode": ep},
        partitioner={
            "decode": [
                CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("decode")]
                )
            ],
        },
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )


def _capture_pass_logs(fn):
    """Run fn and return replacement pass log messages."""
    pass_logger = logging.getLogger("executorch.backends.cuda.triton.replacement_pass")
    prev_level = pass_logger.level
    pass_logger.setLevel(logging.INFO)
    messages = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    pass_logger.addHandler(handler)
    try:
        return fn(), messages
    finally:
        pass_logger.removeHandler(handler)
        pass_logger.setLevel(prev_level)


class TestSplitKReplacement(unittest.TestCase):

    def setUp(self):
        _require_cuda(self)

    def test_large_kv_cache_uses_splitk(self):
        """L_kv=4096 > threshold → split-K selected for decode."""
        model = SDPAModule(n_heads=4, n_kv_heads=2, head_dim=64, kv_len=4096).to(
            torch.bfloat16
        )
        args = (
            torch.zeros(1, 1, 256, dtype=torch.bfloat16),
            torch.tensor([0], dtype=torch.long),
        )

        _, msgs = _capture_pass_logs(lambda: _export_through_cuda_backend(model, args))

        splitk = [m for m in msgs if "split-K" in m]
        self.assertEqual(len(splitk), 1, f"Expected 1 split-K selection. Log: {msgs}")
        self.assertIn("L_kv=4096", splitk[0])

    def test_small_kv_cache_uses_standard(self):
        """L_kv=512 <= threshold → standard SDPA, no split-K."""
        model = SDPAModule(n_heads=4, n_kv_heads=2, head_dim=64, kv_len=512).to(
            torch.bfloat16
        )
        args = (
            torch.zeros(1, 1, 256, dtype=torch.bfloat16),
            torch.tensor([0], dtype=torch.long),
        )

        _, msgs = _capture_pass_logs(lambda: _export_through_cuda_backend(model, args))

        splitk = [m for m in msgs if "split-K" in m]
        self.assertEqual(len(splitk), 0, f"Expected no split-K. Got: {splitk}")

        replaced = [m for m in msgs if "Replaced" in m]
        self.assertTrue(
            any("1 nodes" in m for m in replaced),
            f"Expected 1 SDPA replaced with standard kernel. Log: {msgs}",
        )

    def test_non_pow2_head_dim_uses_standard(self):
        """Non-power-of-2 head_dim → standard SDPA even with large L_kv."""
        model = SDPAModule(n_heads=4, n_kv_heads=2, head_dim=96, kv_len=8192).to(
            torch.bfloat16
        )
        args = (
            torch.zeros(1, 1, 384, dtype=torch.bfloat16),
            torch.tensor([0], dtype=torch.long),
        )

        _, msgs = _capture_pass_logs(lambda: _export_through_cuda_backend(model, args))

        splitk = [m for m in msgs if "split-K" in m]
        self.assertEqual(len(splitk), 0, f"Expected no split-K for D=96. Got: {splitk}")


if __name__ == "__main__":
    unittest.main()
