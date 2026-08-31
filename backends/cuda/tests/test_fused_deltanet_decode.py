# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Correctness test: fully-fused Triton GatedDeltaNet decode kernel vs PyTorch reference.

Verifies that torch.ops.triton.fused_deltanet_decode produces the same output
and state as the original GatedDeltaNet T=1 recurrence with manual Q/K/V split,
L2 norm, head repeat, and gating.
"""

import unittest

import torch
import torch.nn.functional as F


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    if not torch.cuda.is_bf16_supported():
        raise unittest.SkipTest("BF16 not supported on this GPU")


def _import_fused_deltanet_decode():
    from executorch.backends.cuda.triton.kernels.fused_deltanet_decode import (
        fused_deltanet_decode,  # noqa: F401 — registers torch.ops.triton.*
    )

    return fused_deltanet_decode


def _max_abs_error(a, b):
    return (a.float() - b.float()).abs().max().item()


# bf16 kernel vs fp32 reference tolerance.
MAX_ABS_TOL = 0.05
MULTISTEP_TOL = 0.1


def _reference_deltanet_decode(
    qkv_conv,
    alpha,
    beta_raw,
    neg_A_exp,
    dt_bias,
    state,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
):
    """Reference PyTorch implementation matching model.py's original T=1 path.

    Does Q/K/V split, L2 norm, head repeat, gating, then recurrent update.
    """
    B = qkv_conv.shape[0]
    key_dim = num_k_heads * head_k_dim

    q = qkv_conv[:, :key_dim].reshape(B, num_k_heads, head_k_dim)
    k = qkv_conv[:, key_dim : 2 * key_dim].reshape(B, num_k_heads, head_k_dim)
    v = qkv_conv[:, 2 * key_dim :].reshape(B, num_v_heads, head_v_dim)

    q = F.normalize(q.float(), p=2, dim=-1)
    k = F.normalize(k.float(), p=2, dim=-1)
    v = v.float()

    head_repeat = num_v_heads // num_k_heads
    if head_repeat > 1:
        q = q.repeat_interleave(head_repeat, dim=1)
        k = k.repeat_interleave(head_repeat, dim=1)

    beta = torch.sigmoid(beta_raw.float())
    g = neg_A_exp.float() * F.softplus(alpha.float() + dt_bias.float())

    scale = head_k_dim**-0.5
    state_f32 = state.float()

    decay = torch.exp(g).unsqueeze(-1).unsqueeze(-1)
    state_f32 = state_f32 * decay

    Sk = torch.einsum("bhkv,bhk->bhv", state_f32, k)
    delta = beta.unsqueeze(-1) * (v - Sk)
    state_f32 = state_f32 + torch.einsum("bhk,bhv->bhkv", k, delta)

    output = torch.einsum("bhkv,bhk->bhv", state_f32, q) * scale

    new_state = state_f32.to(state.dtype)
    return output, new_state


# Qwen3.5 MoE dimensions (used across tests)
NUM_K_HEADS = 16
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM  # 2048
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM  # 4096
CONV_DIM = 2 * KEY_DIM + VALUE_DIM  # 8192


class TestFusedDeltanetDecode(unittest.TestCase):
    """Test fused GatedDeltaNet decode kernel correctness against PyTorch reference."""

    @classmethod
    def setUpClass(cls):
        _skip_if_no_cuda()
        cls.fused_fn = _import_fused_deltanet_decode()
        torch.manual_seed(42)

        cls.A_log = torch.log(torch.empty(NUM_V_HEADS, device="cuda").uniform_(0.5, 8))
        cls.neg_A_exp = -torch.exp(cls.A_log).float()
        cls.dt_bias = torch.ones(NUM_V_HEADS, device="cuda", dtype=torch.float32)

    def _run_fused(self, qkv, alpha, beta_raw, state):
        """Run fused kernel and return (output, new_state)."""
        output, new_state = torch.ops.triton.fused_deltanet_decode(
            qkv,
            alpha,
            beta_raw,
            self.A_log,
            self.dt_bias,
            state,
        )
        return output, new_state

    def _run_reference(self, qkv, alpha, beta_raw, state):
        """Run reference and return (output, new_state)."""
        return _reference_deltanet_decode(
            qkv,
            alpha,
            beta_raw,
            self.neg_A_exp,
            self.dt_bias,
            state,
            NUM_K_HEADS,
            NUM_V_HEADS,
            HEAD_K_DIM,
            HEAD_V_DIM,
        )

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    def test_basic(self):
        """Single batch, Qwen3.5 MoE dimensions."""
        B = 1
        torch.manual_seed(42)
        qkv = torch.randn(B, CONV_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
        alpha = torch.randn(B, NUM_V_HEADS, device="cuda", dtype=torch.float32)
        beta_raw = torch.randn(B, NUM_V_HEADS, device="cuda", dtype=torch.float32)
        state = (
            torch.randn(
                B,
                NUM_V_HEADS,
                HEAD_K_DIM,
                HEAD_V_DIM,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.1
        )

        ref_out, ref_state = self._run_reference(
            qkv.clone(),
            alpha.clone(),
            beta_raw.clone(),
            state.clone(),
        )
        fused_out, fused_state = self._run_fused(
            qkv.clone(),
            alpha.clone(),
            beta_raw.clone(),
            state.clone(),
        )

        self.assertLess(
            _max_abs_error(fused_out, ref_out), MAX_ABS_TOL, "output mismatch"
        )
        self.assertLess(
            _max_abs_error(fused_state, ref_state), MAX_ABS_TOL, "state mismatch"
        )

    def test_batch(self):
        """Batch size > 1."""
        for B in [2, 4]:
            with self.subTest(B=B):
                torch.manual_seed(42)
                qkv = (
                    torch.randn(B, CONV_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
                )
                alpha = torch.randn(B, NUM_V_HEADS, device="cuda", dtype=torch.float32)
                beta_raw = torch.randn(
                    B, NUM_V_HEADS, device="cuda", dtype=torch.float32
                )
                state = (
                    torch.randn(
                        B,
                        NUM_V_HEADS,
                        HEAD_K_DIM,
                        HEAD_V_DIM,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    * 0.1
                )

                ref_out, ref_state = self._run_reference(
                    qkv.clone(),
                    alpha.clone(),
                    beta_raw.clone(),
                    state.clone(),
                )
                fused_out, fused_state = self._run_fused(
                    qkv.clone(),
                    alpha.clone(),
                    beta_raw.clone(),
                    state.clone(),
                )

                self.assertLess(
                    _max_abs_error(fused_out, ref_out),
                    MAX_ABS_TOL,
                    f"B={B} output mismatch",
                )
                self.assertLess(
                    _max_abs_error(fused_state, ref_state),
                    MAX_ABS_TOL,
                    f"B={B} state mismatch",
                )

    def test_multistep(self):
        """10-step sequential decode checks accumulation drift."""
        torch.manual_seed(42)
        state_ref = (
            torch.randn(
                1,
                NUM_V_HEADS,
                HEAD_K_DIM,
                HEAD_V_DIM,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        state_fused = state_ref.clone()

        for _ in range(10):
            qkv = torch.randn(1, CONV_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
            alpha = torch.randn(1, NUM_V_HEADS, device="cuda", dtype=torch.float32)
            beta_raw = torch.randn(1, NUM_V_HEADS, device="cuda", dtype=torch.float32)

            ref_out, state_ref = self._run_reference(
                qkv.clone(),
                alpha.clone(),
                beta_raw.clone(),
                state_ref,
            )
            fused_out, state_fused = self._run_fused(
                qkv.clone(),
                alpha.clone(),
                beta_raw.clone(),
                state_fused,
            )

        self.assertLess(
            _max_abs_error(fused_out, ref_out),
            MULTISTEP_TOL,
            "multi-step output drift",
        )
        self.assertLess(
            _max_abs_error(state_fused, state_ref),
            MULTISTEP_TOL,
            "multi-step state drift",
        )

    def test_state_not_mutated(self):
        """Kernel must not mutate the input state tensor."""
        B = 1
        torch.manual_seed(42)
        qkv = torch.randn(B, CONV_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
        alpha = torch.randn(B, NUM_V_HEADS, device="cuda", dtype=torch.float32)
        beta_raw = torch.randn(B, NUM_V_HEADS, device="cuda", dtype=torch.float32)
        state = (
            torch.randn(
                B,
                NUM_V_HEADS,
                HEAD_K_DIM,
                HEAD_V_DIM,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.1
        )
        state_copy = state.clone()

        _, _ = self._run_fused(qkv, alpha, beta_raw, state)

        self.assertTrue(torch.equal(state, state_copy), "input state was mutated")

    # ------------------------------------------------------------------
    # CUDA Graph compatibility
    # ------------------------------------------------------------------

    def test_cuda_graph(self):
        """Kernel must be capturable in a CUDA graph."""
        B = 1
        torch.manual_seed(42)
        qkv = torch.randn(B, CONV_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
        alpha = torch.randn(B, NUM_V_HEADS, device="cuda", dtype=torch.float32)
        beta_raw = torch.randn(B, NUM_V_HEADS, device="cuda", dtype=torch.float32)
        state = (
            torch.randn(
                B,
                NUM_V_HEADS,
                HEAD_K_DIM,
                HEAD_V_DIM,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.1
        )

        # Warmup
        for _ in range(3):
            _ = self._run_fused(qkv, alpha, beta_raw, state)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_cg, state_cg = self._run_fused(qkv, alpha, beta_raw, state)

        # Replay
        graph.replay()

        # Compare with reference
        ref_out, _ = self._run_reference(
            qkv.clone(),
            alpha.clone(),
            beta_raw.clone(),
            state.clone(),
        )
        self.assertFalse(torch.isnan(out_cg).any(), "NaN in CUDA graph output")
        self.assertLess(
            _max_abs_error(out_cg, ref_out),
            MAX_ABS_TOL,
            "CUDA graph output mismatch",
        )


if __name__ == "__main__":
    unittest.main()
