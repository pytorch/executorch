#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for mlx::gated_delta_rule custom op + pattern handler.

Usage:
    # Run all configs:
    python -m executorch.backends.mlx.model_ops.test_gated_delta_rule run

    # Run with verbose output:
    python -m executorch.backends.mlx.model_ops.test_gated_delta_rule run -v

    # Rebuild C++ runner first:
    python -m executorch.backends.mlx.model_ops.test_gated_delta_rule run --rebuild
"""

from typing import List, Tuple

import executorch.backends.mlx.model_ops.gated_delta_rule  # noqa: F401

import torch
import torch.nn as nn

from executorch.backends.mlx.test.test_utils import OpTestCase


class GatedDeltaRuleModel(nn.Module):
    """Model using mlx::gated_delta_rule for sequential recurrence."""

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        value_dim: int,
        use_custom_kernel: bool = True,
    ):
        super().__init__()
        self.use_custom_kernel = use_custom_kernel
        self.register_buffer(
            "state",
            torch.zeros(batch_size, num_heads, value_dim, head_dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.mlx.gated_delta_rule(
            q, k, v, g, beta, self.state, use_custom_kernel=self.use_custom_kernel
        )


class GatedDeltaRuleGQAModel(nn.Module):
    """Model with Hk < Hv (GQA-style), matching real Qwen 3.5 config.

    Q and K have num_k_heads heads, V has num_v_heads heads.
    Q and K are repeat_interleaved to match num_v_heads before the custom op call,
    matching the pattern in _exportable_gated_delta_net_forward.
    """

    def __init__(
        self,
        batch_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_dim: int,
        value_dim: int,
        use_custom_kernel: bool = True,
    ):
        super().__init__()
        assert num_v_heads % num_k_heads == 0
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_repeat = num_v_heads // num_k_heads
        self.use_custom_kernel = use_custom_kernel
        self.register_buffer(
            "state",
            torch.zeros(batch_size, num_v_heads, value_dim, head_dim),
        )

    def forward(
        self,
        q: torch.Tensor,  # [B, T, Hk, Dk]
        k: torch.Tensor,  # [B, T, Hk, Dk]
        v: torch.Tensor,  # [B, T, Hv, Dv]
        g: torch.Tensor,  # [B, T, Hv]
        beta: torch.Tensor,  # [B, T, Hv]
    ) -> torch.Tensor:
        if self.head_repeat > 1:
            q = q.repeat_interleave(self.head_repeat, dim=2)
            k = k.repeat_interleave(self.head_repeat, dim=2)
        return torch.ops.mlx.gated_delta_rule(
            q, k, v, g, beta, self.state, use_custom_kernel=self.use_custom_kernel
        )


class GatedDeltaRuleMultiStepModel(nn.Module):
    """Model that calls gated_delta_rule TWICE to test state carry-forward.

    The second call reads the state mutated by the first call.
    If state doesn't persist, out2 would be identical to running with
    zero state — which is wrong.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        value_dim: int,
        use_custom_kernel: bool = False,
    ):
        super().__init__()
        self.use_custom_kernel = use_custom_kernel
        self.register_buffer(
            "state",
            torch.zeros(batch_size, num_heads, value_dim, head_dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        # Step 1: process inputs, mutates self.state
        out1 = torch.ops.mlx.gated_delta_rule(
            q, k, v, g, beta, self.state, use_custom_kernel=self.use_custom_kernel
        )
        # Step 2: same inputs, but state carries from step 1
        out2 = torch.ops.mlx.gated_delta_rule(
            q, k, v, g, beta, self.state, use_custom_kernel=self.use_custom_kernel
        )
        # Return concatenated so we can verify both outputs
        return torch.cat([out1, out2], dim=1)


class GatedDeltaRuleTest(OpTestCase):
    """Test case for mlx::gated_delta_rule (ScanNode and MetalKernelNode)."""

    name = "gated_delta_rule"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        batch_size: int = 1,
        seq_len: int = 4,
        num_heads: int = 2,
        head_dim: int = 16,
        value_dim: int = 16,
        dtype: torch.dtype = torch.float32,
        rtol: float = 1e-4,
        atol: float = 1e-4,
        use_custom_kernel: bool = False,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.dtype = dtype
        self.rtol = rtol
        self.atol = atol
        self.use_custom_kernel = use_custom_kernel

        parts = [
            "gated_delta_rule",
            f"b{batch_size}",
            f"t{seq_len}",
            f"h{num_heads}",
            f"dk{head_dim}",
            f"dv{value_dim}",
        ]
        if dtype != torch.float32:
            parts.append(str(dtype).split(".")[-1])
        parts.append("kernel" if use_custom_kernel else "scan")
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["GatedDeltaRuleTest"]:
        configs = []
        # Small dims (Dk not multiple of 32) — scan-only
        for use_kernel in [False]:
            configs.append(cls(use_custom_kernel=use_kernel))
            configs.append(cls(seq_len=1, use_custom_kernel=use_kernel))
            configs.append(
                cls(
                    seq_len=8,
                    num_heads=4,
                    head_dim=32,
                    value_dim=32,
                    use_custom_kernel=use_kernel,
                )
            )
            configs.append(
                cls(
                    dtype=torch.bfloat16,
                    rtol=0.05,
                    atol=0.15,
                    use_custom_kernel=use_kernel,
                )
            )
        # Dims with Dk multiple of 32 — both scan and custom kernel
        for use_kernel in [False, True]:
            configs.append(
                cls(
                    num_heads=2, head_dim=64, value_dim=64, use_custom_kernel=use_kernel
                )
            )
            configs.append(
                cls(
                    seq_len=1,
                    num_heads=2,
                    head_dim=64,
                    value_dim=64,
                    use_custom_kernel=use_kernel,
                )
            )
            configs.append(
                cls(
                    seq_len=8,
                    num_heads=4,
                    head_dim=64,
                    value_dim=64,
                    use_custom_kernel=use_kernel,
                )
            )
            configs.append(
                cls(
                    num_heads=2,
                    head_dim=64,
                    value_dim=64,
                    dtype=torch.bfloat16,
                    rtol=0.05,
                    atol=0.15,
                    use_custom_kernel=use_kernel,
                )
            )
            configs.append(
                cls(
                    num_heads=2,
                    head_dim=128,
                    value_dim=128,
                    use_custom_kernel=use_kernel,
                )
            )
        return configs

    def create_model(self) -> nn.Module:
        model = GatedDeltaRuleModel(
            self.batch_size,
            self.num_heads,
            self.head_dim,
            self.value_dim,
            use_custom_kernel=self.use_custom_kernel,
        )
        return model.to(self.dtype)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        # Scale q and k by 1/√Dk to keep dot products in a reasonable range.
        # Without this, bf16 accumulation diverges at larger head dims (dk64+)
        # because sum-of-64 products grows to ~O(√Dk) per step, compounding
        # exponentially through the recurrence.
        scale = self.head_dim**-0.5
        q = (
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
            )
            * scale
        )
        k = (
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
            )
            * scale
        )
        v = torch.randn(
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.value_dim,
            dtype=self.dtype,
        )
        g = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, dtype=self.dtype
        ).sigmoid()
        beta = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, dtype=self.dtype
        ).sigmoid()
        return (q, k, v, g, beta)


class GatedDeltaRuleDynamicSeqTest(OpTestCase):
    """Test gated_delta_rule with dynamic seq_len.

    Exports with seq_len=export_seq_len using dynamic shapes, then runs
    inference with seq_len=test_seq_len. Verifies the MetalKernelNode/ScanNode
    handles runtime sequence lengths different from the trace-time value.
    """

    name = "gated_delta_rule_dynamic"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        export_seq_len: int = 4,
        test_seq_len: int = 1,
        num_heads: int = 2,
        head_dim: int = 64,
        value_dim: int = 64,
        dtype: torch.dtype = torch.float32,
        rtol: float = 1e-4,
        atol: float = 1e-4,
        use_custom_kernel: bool = False,
    ):
        self.batch_size = 1
        self.export_seq_len = export_seq_len
        self.test_seq_len = test_seq_len
        self.seq_len = export_seq_len  # used by create_inputs (export tracing)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.dtype = dtype
        self.rtol = rtol
        self.atol = atol
        self.use_custom_kernel = use_custom_kernel

        parts = [
            "gated_delta_rule_dynamic",
            f"export_t{export_seq_len}",
            f"test_t{test_seq_len}",
            f"h{num_heads}",
            f"dk{head_dim}",
        ]
        parts.append("kernel" if use_custom_kernel else "scan")
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["GatedDeltaRuleDynamicSeqTest"]:
        configs = []
        for use_kernel in [False, True]:
            # Export with T=4, test with T=1 (decode)
            configs.append(
                cls(export_seq_len=4, test_seq_len=1, use_custom_kernel=use_kernel)
            )
            # Export with T=2, test with T=8 (longer prefill)
            configs.append(
                cls(export_seq_len=2, test_seq_len=8, use_custom_kernel=use_kernel)
            )
            # Export with T=4, test with T=4 (same — control)
            configs.append(
                cls(export_seq_len=4, test_seq_len=4, use_custom_kernel=use_kernel)
            )
        return configs

    def get_dynamic_shapes(self):
        # All 5 inputs (q, k, v, g, beta) have seq_len at dim 1
        seq_dim = torch.export.Dim("seq_len", min=1, max=32)
        return {
            "q": {1: seq_dim},
            "k": {1: seq_dim},
            "v": {1: seq_dim},
            "g": {1: seq_dim},
            "beta": {1: seq_dim},
        }

    def create_model(self) -> nn.Module:
        model = GatedDeltaRuleModel(
            self.batch_size,
            self.num_heads,
            self.head_dim,
            self.value_dim,
            use_custom_kernel=self.use_custom_kernel,
        )
        return model.to(self.dtype)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Inputs for export tracing (uses export_seq_len)."""
        scale = self.head_dim**-0.5
        T = self.export_seq_len
        q = (
            torch.randn(
                self.batch_size, T, self.num_heads, self.head_dim, dtype=self.dtype
            )
            * scale
        )
        k = (
            torch.randn(
                self.batch_size, T, self.num_heads, self.head_dim, dtype=self.dtype
            )
            * scale
        )
        v = torch.randn(
            self.batch_size, T, self.num_heads, self.value_dim, dtype=self.dtype
        )
        g = torch.randn(self.batch_size, T, self.num_heads, dtype=self.dtype).sigmoid()
        beta = torch.randn(
            self.batch_size, T, self.num_heads, dtype=self.dtype
        ).sigmoid()
        return (q, k, v, g, beta)

    def create_test_inputs(self) -> Tuple[torch.Tensor, ...]:
        """Inputs for runtime test (uses test_seq_len — may differ from export)."""
        scale = self.head_dim**-0.5
        T = self.test_seq_len
        q = (
            torch.randn(
                self.batch_size, T, self.num_heads, self.head_dim, dtype=self.dtype
            )
            * scale
        )
        k = (
            torch.randn(
                self.batch_size, T, self.num_heads, self.head_dim, dtype=self.dtype
            )
            * scale
        )
        v = torch.randn(
            self.batch_size, T, self.num_heads, self.value_dim, dtype=self.dtype
        )
        g = torch.randn(self.batch_size, T, self.num_heads, dtype=self.dtype).sigmoid()
        beta = torch.randn(
            self.batch_size, T, self.num_heads, dtype=self.dtype
        ).sigmoid()
        return (q, k, v, g, beta)


class GatedDeltaRuleGQATest(OpTestCase):
    """Test gated_delta_rule with Hk < Hv (GQA-style head repeat).

    Matches real Qwen 3.5 config where num_k_heads=1, num_v_heads=2 (tiny)
    or num_k_heads=8, num_v_heads=64 (full). Q and K get repeat_interleaved
    before the custom op call.
    """

    name = "gated_delta_rule_gqa"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        batch_size: int = 1,
        seq_len: int = 4,
        num_k_heads: int = 1,
        num_v_heads: int = 2,
        head_dim: int = 64,
        value_dim: int = 64,
        dtype: torch.dtype = torch.float32,
        rtol: float = 1e-4,
        atol: float = 1e-4,
        use_custom_kernel: bool = False,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.dtype = dtype
        self.rtol = rtol
        self.atol = atol
        self.use_custom_kernel = use_custom_kernel

        parts = [
            "gated_delta_rule_gqa",
            f"b{batch_size}",
            f"t{seq_len}",
            f"hk{num_k_heads}",
            f"hv{num_v_heads}",
            f"dk{head_dim}",
            f"dv{value_dim}",
        ]
        if dtype != torch.float32:
            parts.append(str(dtype).split(".")[-1])
        parts.append("kernel" if use_custom_kernel else "scan")
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["GatedDeltaRuleGQATest"]:
        configs = []
        for use_kernel in [False, True]:
            # Tiny config: Hk=1, Hv=2 (head_repeat=2)
            configs.append(
                cls(num_k_heads=1, num_v_heads=2, use_custom_kernel=use_kernel)
            )
            # Decode (T=1) with GQA
            configs.append(
                cls(
                    seq_len=1,
                    num_k_heads=1,
                    num_v_heads=2,
                    use_custom_kernel=use_kernel,
                )
            )
            # Larger head ratio: Hk=2, Hv=8 (head_repeat=4)
            configs.append(
                cls(num_k_heads=2, num_v_heads=8, use_custom_kernel=use_kernel)
            )
            # bf16 with GQA
            configs.append(
                cls(
                    num_k_heads=1,
                    num_v_heads=2,
                    dtype=torch.bfloat16,
                    rtol=0.05,
                    atol=0.15,
                    use_custom_kernel=use_kernel,
                )
            )
        return configs

    def create_model(self) -> nn.Module:
        model = GatedDeltaRuleGQAModel(
            self.batch_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_dim,
            self.value_dim,
            use_custom_kernel=self.use_custom_kernel,
        )
        return model.to(self.dtype)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        scale = self.head_dim**-0.5
        # Q and K have num_k_heads (model does repeat_interleave internally)
        q = (
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_k_heads,
                self.head_dim,
                dtype=self.dtype,
            )
            * scale
        )
        k = (
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_k_heads,
                self.head_dim,
                dtype=self.dtype,
            )
            * scale
        )
        # V, g, beta have num_v_heads
        v = torch.randn(
            self.batch_size,
            self.seq_len,
            self.num_v_heads,
            self.value_dim,
            dtype=self.dtype,
        )
        g = torch.randn(
            self.batch_size, self.seq_len, self.num_v_heads, dtype=self.dtype
        ).sigmoid()
        beta = torch.randn(
            self.batch_size, self.seq_len, self.num_v_heads, dtype=self.dtype
        ).sigmoid()
        return (q, k, v, g, beta)


class GatedDeltaRuleFloatCastModel(nn.Module):
    """Model that mirrors the export pattern: bf16 inputs, fp32 state buffer.

    The recurrent state must be fp32 for numerical stability. Rather than
    casting with .float() (which creates a temporary that breaks mutation
    tracking), the state buffer is registered as fp32 from the start.
    Inputs are cast to fp32 before the op call.
    """

    def __init__(self, batch_size: int, num_heads: int, head_dim: int, value_dim: int):
        super().__init__()
        # fp32 state buffer — NOT bf16. Avoids .float() cast that breaks mutation.
        self.register_buffer(
            "state",
            torch.zeros(
                batch_size, num_heads, value_dim, head_dim, dtype=torch.float32
            ),
        )

    def forward(
        self,
        q: torch.Tensor,  # bf16
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.ops.mlx.gated_delta_rule(
            q.float(),
            k.float(),
            v.float(),
            g.float(),
            beta.float(),
            self.state,  # already fp32, no cast needed
            use_custom_kernel=False,
        )
        return output.to(q.dtype)


class GatedDeltaRuleFloatCastTest(OpTestCase):
    """Test gated_delta_rule with bf16 state + fp32 cast (mirrors export model)."""

    name = "gated_delta_rule_float_cast"
    rtol = 0.05
    atol = 0.15

    @classmethod
    def get_test_configs(cls) -> List["GatedDeltaRuleFloatCastTest"]:
        return [cls()]

    def __init__(self):
        self.batch_size = 1
        self.seq_len = 4
        self.num_heads = 2
        self.head_dim = 16
        self.value_dim = 16
        self.dtype = torch.bfloat16
        self.name = "gated_delta_rule_float_cast_b1_t4_h2"

    def create_model(self) -> nn.Module:
        return GatedDeltaRuleFloatCastModel(
            self.batch_size,
            self.num_heads,
            self.head_dim,
            self.value_dim,
        )

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        q = torch.randn(
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
        )
        k = torch.randn(
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
        )
        v = torch.randn(
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.value_dim,
            dtype=self.dtype,
        )
        g = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, dtype=self.dtype
        ).sigmoid()
        beta = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, dtype=self.dtype
        ).sigmoid()
        return (q, k, v, g, beta)


class GatedDeltaRuleWithProjectionModel(nn.Module):
    """Model that derives Q, K, V from a shared projection, mimicking the real export.

    Uses bf16 weights with fp32 state buffer. The .float() casts on Q, K, V,
    g, beta create intermediate ASTYPE temp slots that the delete-as-you-go
    allocator can free and reuse, potentially causing Q and K to share the
    same slot in the ScanNode originals.
    """

    def __init__(self, batch_size: int, num_heads: int, head_dim: int, value_dim: int):
        super().__init__()
        qkv_dim = 2 * num_heads * head_dim + num_heads * value_dim
        gate_dim = num_heads  # g and beta each have num_heads dims
        self.proj = nn.Linear(num_heads * head_dim, qkv_dim + 2 * gate_dim, bias=False)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = value_dim
        # fp32 state buffer (same as export model)
        self.register_buffer(
            "state",
            torch.zeros(
                batch_size, num_heads, value_dim, head_dim, dtype=torch.float32
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        nh = self.num_heads
        hd = self.head_dim
        vd = self.value_dim

        proj = self.proj(x)  # bf16
        kd = nh * hd

        # Split into Q, K, V, g, beta — same pattern as the real GDN forward
        q = proj[..., :kd].reshape(B, T, nh, hd)
        k = proj[..., kd : 2 * kd].reshape(B, T, nh, hd)
        v = proj[..., 2 * kd : 2 * kd + nh * vd].reshape(B, T, nh, vd)
        g = proj[..., 2 * kd + nh * vd : 2 * kd + nh * vd + nh].sigmoid()
        beta = proj[..., 2 * kd + nh * vd + nh :].sigmoid()

        # L2-normalize Q and K (creates intermediate temps)
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

        # .float() casts create ASTYPE temp slots that can alias
        output = torch.ops.mlx.gated_delta_rule(
            q.float(),
            k.float(),
            v.float(),
            g.float(),
            beta.float(),
            self.state,  # already fp32
            use_custom_kernel=False,
        )
        return output.to(x.dtype)


class GatedDeltaRuleWithProjectionTest(OpTestCase):
    """Test gated_delta_rule with shared projection → normalize → float cast.

    This reproduces the slot aliasing bug where Q and K get the same temp slot
    because the delete-as-you-go allocator frees Q's float cast slot before K's
    float cast allocates one.
    """

    name = "gated_delta_rule_projection"
    rtol = 1e-4
    atol = 1e-4

    @classmethod
    def get_test_configs(cls) -> List["GatedDeltaRuleWithProjectionTest"]:
        return [cls()]

    def __init__(self):
        self.batch_size = 1
        self.seq_len = 4
        self.num_heads = 2
        self.head_dim = 16
        self.value_dim = 16
        self.dtype = torch.bfloat16  # bf16 so .float() casts produce ASTYPE temp slots
        self.rtol = 0.05
        self.atol = 0.15
        self.name = "gated_delta_rule_projection_b1_t4_h2"

    def create_model(self) -> nn.Module:
        model = GatedDeltaRuleWithProjectionModel(
            self.batch_size,
            self.num_heads,
            self.head_dim,
            self.value_dim,
        )
        return model.to(torch.bfloat16)  # bf16 weights so .float() casts are real

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        x = torch.randn(
            self.batch_size,
            self.seq_len,
            self.num_heads * self.head_dim,
            dtype=torch.bfloat16,
        )
        return (x,)


class GatedDeltaRuleMultiStepTest(OpTestCase):
    """Test that state carries forward between two gated_delta_rule calls.

    Uses GatedDeltaRuleMultiStepModel which calls the op twice in a single
    forward. The second call reads the mutated state from the first. If state
    doesn't persist, out2 would equal running with zero state — and the
    concatenated output would mismatch the eager reference.
    """

    name = "gated_delta_rule_multistep"
    rtol = 1e-4
    atol = 1e-4

    def __init__(
        self,
        batch_size: int = 1,
        seq_len: int = 4,
        num_heads: int = 2,
        head_dim: int = 16,
        value_dim: int = 16,
        dtype: torch.dtype = torch.float32,
        rtol: float = 1e-4,
        atol: float = 1e-4,
        use_custom_kernel: bool = False,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.dtype = dtype
        self.rtol = rtol
        self.atol = atol
        self.use_custom_kernel = use_custom_kernel

        parts = [
            "gated_delta_rule_multistep",
            f"b{batch_size}",
            f"t{seq_len}",
            f"h{num_heads}",
            f"dk{head_dim}",
            f"dv{value_dim}",
        ]
        if dtype != torch.float32:
            parts.append(str(dtype).split(".")[-1])
        parts.append("kernel" if use_custom_kernel else "scan")
        self.name = "_".join(parts)

    @classmethod
    def get_test_configs(cls) -> List["GatedDeltaRuleMultiStepTest"]:
        return [
            cls(),
            cls(seq_len=1),
            # Dk=64 with custom kernel — exposes [int8] serialization bug
            # if state_out dtype is corrupted to u8
            cls(num_heads=2, head_dim=64, value_dim=64, use_custom_kernel=True),
            cls(
                seq_len=1,
                num_heads=2,
                head_dim=64,
                value_dim=64,
                use_custom_kernel=True,
            ),
            # bf16 multistep with kernel — tests precision over two calls
            cls(
                num_heads=2,
                head_dim=64,
                value_dim=64,
                dtype=torch.bfloat16,
                rtol=0.05,
                atol=0.15,
                use_custom_kernel=True,
            ),
        ]

    def create_model(self) -> nn.Module:
        model = GatedDeltaRuleMultiStepModel(
            self.batch_size,
            self.num_heads,
            self.head_dim,
            self.value_dim,
            use_custom_kernel=self.use_custom_kernel,
        )
        return model.to(self.dtype)

    def create_inputs(self) -> Tuple[torch.Tensor, ...]:
        scale = self.head_dim**-0.5
        q = (
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
            )
            * scale
        )
        k = (
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
            )
            * scale
        )
        v = torch.randn(
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.value_dim,
            dtype=self.dtype,
        )
        g = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, dtype=self.dtype
        ).sigmoid()
        beta = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, dtype=self.dtype
        ).sigmoid()
        return (q, k, v, g, beta)


if __name__ == "__main__":  # noqa: C901
    import argparse
    import sys

    from executorch.backends.mlx.test.test_utils import rebuild_op_test_runner

    parser = argparse.ArgumentParser(
        description="Test mlx::gated_delta_rule op (ScanNode)"
    )
    parser.add_argument(
        "action",
        choices=["generate", "compare", "run", "list"],
        help="Action: generate (export), compare (check outputs), run (full test), list (show configs)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild C++ runner first"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Run specific config by name"
    )
    args = parser.parse_args()

    if args.rebuild:
        if not rebuild_op_test_runner(verbose=args.verbose):
            sys.exit(1)

    configs = (
        GatedDeltaRuleTest.get_test_configs()
        + GatedDeltaRuleDynamicSeqTest.get_test_configs()
        + GatedDeltaRuleGQATest.get_test_configs()
        + GatedDeltaRuleFloatCastTest.get_test_configs()
        + GatedDeltaRuleWithProjectionTest.get_test_configs()
        + GatedDeltaRuleMultiStepTest.get_test_configs()
    )

    if args.action == "list":
        for cfg in configs:
            print(f"  {cfg.name}")
        sys.exit(0)

    if args.config:
        configs = [c for c in configs if c.name == args.config]
        if not configs:
            print(f"No config matching '{args.config}'")
            sys.exit(1)

    passed = 0
    failed = 0
    failed_names = []

    for test in configs:
        if args.action == "generate":
            pte_path, input_path, expected_path = test.generate_test_files(
                verbose=args.verbose
            )
            print(f"Generated: {pte_path}")
        elif args.action == "compare":
            actual_path = test.get_test_dir() / "actual_output.bin"
            ok, msg = test.compare_with_actual(actual_path)
            print(f"{'✓' if ok else '✗'} {test.name}: {msg}")
            if ok:
                passed += 1
            else:
                failed += 1
                failed_names.append(test.name)
        elif args.action == "run":
            ok = test.run_test(verbose=args.verbose)
            if ok:
                passed += 1
            else:
                failed += 1
                failed_names.append(test.name)

    if args.action in ("run", "compare"):
        print(f"\nPassed: {passed}, Failed: {failed}")
        if failed_names:
            print(f"Failed: {', '.join(failed_names)}")
        sys.exit(0 if failed == 0 else 1)
