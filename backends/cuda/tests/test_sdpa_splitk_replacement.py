# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test which SDPA calls ReplaceEdgeOpWithTritonOpPass replaces, and with which kernel.

Exports minimal models containing F.scaled_dot_product_attention through the
CUDA backend. CUDA routes eligible decode shapes to split-K, ROCm and other
shapes use standard SDPA, and calls neither kernel accepts stay with the
regular lowering.
"""

import importlib.util
import logging
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F


def _require_cuda(tc: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        tc.skipTest("CUDA required")


def _require_splitk(tc: unittest.TestCase) -> None:
    if torch.version.hip is not None:
        tc.skipTest("split-K is off on ROCm")


def _require_regular_lowering_on_rocm(tc: unittest.TestCase) -> None:
    # On ROCm, AOTInductor's C++ wrapper fails to compile the regular lowering of
    # some SDPA calls: a generated kernel's argument types and names differ.
    # TODO: remove once https://github.com/pytorch/pytorch/issues/199619 is fixed.
    if torch.version.hip is not None:
        tc.skipTest("AOTInductor cannot lower this SDPA call on ROCm")


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

    def test_below_threshold_uses_standard(self):
        """L_kv=128 < threshold (256) -> standard SDPA, no split-K."""
        model = SDPAModule(n_heads=4, n_kv_heads=2, head_dim=64, kv_len=128).to(
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

    def test_at_threshold_uses_backend_kernel(self):
        """L_kv=256 selects split-K on CUDA and standard SDPA on ROCm."""
        model = SDPAModule(n_heads=4, n_kv_heads=2, head_dim=64, kv_len=256).to(
            torch.bfloat16
        )
        args = (
            torch.zeros(1, 1, 256, dtype=torch.bfloat16),
            torch.tensor([0], dtype=torch.long),
        )

        _, msgs = _capture_pass_logs(lambda: _export_through_cuda_backend(model, args))

        splitk = [m for m in msgs if "split-K" in m]
        expected = 0 if torch.version.hip is not None else 1
        self.assertEqual(
            len(splitk),
            expected,
            f"Expected {expected} split-K selections. Log: {msgs}",
        )
        if expected:
            self.assertIn("L_kv=256", splitk[0])

        replaced = [m for m in msgs if "Replaced" in m]
        self.assertTrue(
            any("1 nodes" in m for m in replaced),
            f"Expected 1 SDPA replaced with a Triton kernel. Log: {msgs}",
        )

    def test_large_kv_cache_uses_backend_kernel(self):
        """L_kv=4096 selects split-K on CUDA and standard SDPA on ROCm."""
        model = SDPAModule(n_heads=4, n_kv_heads=2, head_dim=64, kv_len=4096).to(
            torch.bfloat16
        )
        args = (
            torch.zeros(1, 1, 256, dtype=torch.bfloat16),
            torch.tensor([0], dtype=torch.long),
        )

        _, msgs = _capture_pass_logs(lambda: _export_through_cuda_backend(model, args))

        splitk = [m for m in msgs if "split-K" in m]
        expected = 0 if torch.version.hip is not None else 1
        self.assertEqual(
            len(splitk),
            expected,
            f"Expected {expected} split-K selections. Log: {msgs}",
        )
        if expected:
            self.assertIn("L_kv=4096", splitk[0])

        replaced = [m for m in msgs if "Replaced" in m]
        self.assertTrue(
            any("1 nodes" in m for m in replaced),
            f"Expected 1 SDPA replaced with a Triton kernel. Log: {msgs}",
        )

    def test_non_pow2_head_dim_uses_standard(self):
        """Non-power-of-2 head_dim -> standard SDPA even with large L_kv."""
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


class MaskedSDPAModule(nn.Module):
    def __init__(self, is_causal=False, dropout_p=0.0):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_p = dropout_p

    def forward(self, q, k, v, mask=None):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout_p,
            is_causal=self.is_causal,
        )


def _bf16(*shape):
    return torch.randn(*shape, dtype=torch.bfloat16)


class TestUnsupportedSDPAStaysWithRegularLowering(unittest.TestCase):
    """SDPA calls the Triton kernels do not accept are left to the regular lowering."""

    def setUp(self):
        _require_cuda(self)

    def _logs(self, q, k, v, mask=None, **kwargs):
        model = MaskedSDPAModule(**kwargs)
        args = (q, k, v) if mask is None else (q, k, v, mask)
        _, msgs = _capture_pass_logs(lambda: _export_through_cuda_backend(model, args))
        return msgs

    def assertStays(self, msgs, reason):
        self.assertTrue(any(reason in m for m in msgs), msgs)
        self.assertTrue(any("Replaced 0 nodes" in m for m in msgs), msgs)

    def test_float_mask_is_not_replaced(self):
        _require_regular_lowering_on_rocm(self)
        q = _bf16(1, 2, 8, 16)
        mask = torch.zeros(1, 1, 8, 8, dtype=torch.bfloat16)
        self.assertStays(self._logs(q, q, q, mask), "attn_mask must have dtype")

    def test_float32_inputs_are_not_replaced(self):
        q = torch.randn(1, 2, 8, 16)
        self.assertStays(self._logs(q, q, q), "Expected bfloat16 inputs")

    def test_broadcast_bool_mask_is_not_replaced(self):
        q = _bf16(1, 2, 8, 16)
        mask = torch.ones(1, 1, 1, 8, dtype=torch.bool)
        self.assertStays(self._logs(q, q, q, mask), "attn_mask shape mismatch")

    def test_batch_1_mask_under_batch_2_is_not_replaced(self):
        q = _bf16(2, 2, 8, 16)
        mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        self.assertStays(self._logs(q, q, q, mask), "attn_mask shape mismatch")

    def test_value_head_size_other_than_query_is_not_replaced(self):
        q, v = _bf16(1, 2, 1, 16), _bf16(1, 2, 256, 32)
        self.assertStays(self._logs(q, _bf16(1, 2, 256, 16), v), "Head dimension")

    def test_key_value_batch_1_under_batch_2_is_not_replaced(self):
        q, kv = _bf16(2, 2, 1, 16), _bf16(1, 2, 256, 16)
        self.assertStays(self._logs(q, kv, kv), "Batch dimension must match")

    def test_shared_key_value_head_without_gqa_is_not_replaced(self):
        q, kv = _bf16(1, 4, 8, 16), _bf16(1, 1, 8, 16)
        self.assertStays(self._logs(q, kv, kv), "Head counts must match")

    def test_causal_with_other_lengths_is_not_replaced(self):
        q, kv = _bf16(1, 2, 4, 16), _bf16(1, 2, 8, 16)
        self.assertStays(self._logs(q, kv, kv, is_causal=True), "Causal masking")

    def test_dropout_is_not_replaced(self):
        _require_regular_lowering_on_rocm(self)
        q = _bf16(1, 2, 8, 16)
        self.assertStays(self._logs(q, q, q, dropout_p=0.1), "dropout_p must be 0.0")

    def test_bool_mask_is_still_replaced(self):
        q = _bf16(1, 2, 8, 16)
        mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        msgs = self._logs(q, q, q, mask)
        self.assertTrue(any("Replaced 1 nodes" in m for m in msgs), msgs)

    def assertSplitK(self, msgs):
        self.assertTrue(any("Using split-K decode SDPA" in m for m in msgs), msgs)
        self.assertTrue(any("Replaced 1 nodes" in m for m in msgs), msgs)

    def test_masked_decode_still_uses_splitk(self):
        _require_splitk(self)
        q, kv = _bf16(1, 4, 1, 64), _bf16(1, 4, 512, 64)
        mask = torch.ones(1, 1, 1, 512, dtype=torch.bool)
        self.assertSplitK(self._logs(q, kv, kv, mask))

    def test_causal_decode_still_uses_splitk(self):
        _require_splitk(self)
        q, kv = _bf16(1, 4, 1, 64), _bf16(1, 4, 512, 64)
        self.assertSplitK(self._logs(q, kv, kv, is_causal=True))

    def test_decode_with_a_shared_key_value_head_still_uses_splitk(self):
        _require_splitk(self)
        q, kv = _bf16(1, 4, 1, 64), _bf16(1, 1, 512, 64)
        self.assertSplitK(self._logs(q, kv, kv))


class TestSDPAKernelSupportCheck(unittest.TestCase):
    """Calls that cannot be exported end to end, on hand-built graphs."""

    def setUp(self):
        if importlib.util.find_spec("triton") is None:
            self.skipTest("Triton required")

    def _sdpa_node(self, shape, dropout_p=0.0):
        from executorch.exir.dialects._ops import ops as exir_ops
        from torch._subclasses.fake_tensor import FakeTensorMode

        graph = torch.fx.Graph()
        with FakeTensorMode():
            q = torch.empty(*shape, dtype=torch.bfloat16, device="cuda")
        inputs = []
        for name in ("q", "k", "v"):
            node = graph.placeholder(name)
            node.meta["val"] = q
            inputs.append(node)
        return graph.call_function(
            exir_ops.edge.aten.scaled_dot_product_attention.default,
            (*inputs, None, dropout_p),
        )

    def test_2d_inputs_are_not_supported(self):
        # The regular lowering of a 2D SDPA fails later in Inductor, so this
        # case cannot be exported end to end.
        from executorch.backends.cuda.triton.replacement_pass import (
            ReplaceEdgeOpWithTritonOpPass,
        )

        node = self._sdpa_node((8, 16))
        self.assertFalse(ReplaceEdgeOpWithTritonOpPass._sdpa_kernel_supports(node))

    def test_named_inputs_are_not_supported(self):
        from executorch.backends.cuda.triton.replacement_pass import (
            ReplaceEdgeOpWithTritonOpPass,
        )

        node = self._sdpa_node((1, 2, 8, 16))
        query, key, value = node.args[:3]
        node.args = ()
        node.kwargs = {"query": query, "key": key, "value": value}
        self.assertFalse(ReplaceEdgeOpWithTritonOpPass._sdpa_kernel_supports(node))


if __name__ == "__main__":
    unittest.main()
