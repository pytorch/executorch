# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Non-causal fused SDPA (`et_vk.sdpa.default`) export + reference checks.

This is the SAM2/SigLIP/DaViT/BART attention op (NOT the causal KV-cache
`sdpa_with_kv_cache` covered by `test/ops/test_sdpa.py`). The et_vk source
transform `_et_vk_sdpa_attn` plugs `torch.ops.et_vk.sdpa.default(q, k, v,
attn_mask, scale)` (q/k/v `[B, H, S, D]`) into every Florence-2 attention block;
it is plain non-causal attention `softmax(q @ kᵀ * scale + attn_mask) @ v`.

`test_export_delegates` checks the op lowers into the VulkanBackend delegate.
`test_golden_matches_eager` checks the custom op's eager math matches
`F.scaled_dot_product_attention` (so the on-device golden can't be self-
fulfilling). On-device GPU numerics run via the op-test framework on a GPU rig.
"""

import unittest
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401  registers et_vk.sdpa
from executorch.backends.vulkan import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

NEG_INF = -1e30


@dataclass(frozen=True)
class SdpaConfig:
    name: str
    b: int
    h: int
    s_q: int
    s_kv: int
    d: int
    masked: bool = False
    causal: bool = False


# Shapes from real SAM2/SigLIP/DaViT encoders + a cheap correctness case + an
# asymmetric (S_q != S_kv) pooled-query case + the BART causal-mask path.
CONFIGS = [
    SdpaConfig("selfattn_small", 1, 4, 8, 8, 16),
    SdpaConfig("selfattn_siglip", 1, 12, 576, 576, 64),
    SdpaConfig("asym_qpool", 1, 8, 4, 16, 16),
    SdpaConfig("masked", 1, 4, 8, 8, 16, masked=True),
    SdpaConfig("causal", 1, 4, 8, 8, 16, causal=True),
]


def _qkv(cfg: SdpaConfig):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(cfg.b, cfg.h, cfg.s_q, cfg.d, generator=g)
    k = torch.randn(cfg.b, cfg.h, cfg.s_kv, cfg.d, generator=g)
    v = torch.randn(cfg.b, cfg.h, cfg.s_kv, cfg.d, generator=g)
    return q, k, v


def _mask(cfg: SdpaConfig) -> Optional[torch.Tensor]:
    if cfg.causal:
        assert cfg.s_q == cfg.s_kv, "causal mask requires S_q == S_kv"
        m = torch.triu(
            torch.full((cfg.s_q, cfg.s_kv), NEG_INF, dtype=torch.float32), diagonal=1
        )
        return m.reshape(1, 1, cfg.s_q, cfg.s_kv).expand(
            cfg.b, cfg.h, cfg.s_q, cfg.s_kv
        ).contiguous()
    if not cfg.masked:
        return None
    g = torch.Generator().manual_seed(1)
    return torch.randn(cfg.b, cfg.h, cfg.s_q, cfg.s_kv, generator=g).clamp(-1.0, 0.0)


class SdpaModule(torch.nn.Module):
    """Wraps the registered et_vk.sdpa op. A baked mask is held as a buffer
    (constant) so the partitioner prepacks it; the runner forwards only q/k/v."""

    def __init__(self, mask: Optional[torch.Tensor] = None):
        super().__init__()
        if mask is not None:
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def forward(self, q, k, v):
        return torch.ops.et_vk.sdpa.default(q, k, v, self.mask, None)


def _lower(cfg: SdpaConfig, q, k, v):
    module = SdpaModule(_mask(cfg)).eval()
    ep = torch.export.export(module, (q, k, v))
    return to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])


class TestEtVkSdpa(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for cfg in CONFIGS:
            with self.subTest(config=cfg.name):
                q, k, v = _qkv(cfg)
                et = _lower(cfg, q, k, v).to_executorch()
                found = any(
                    d.id == "VulkanBackend"
                    for plan in et.executorch_program.execution_plan
                    for d in plan.delegates
                )
                self.assertTrue(
                    found, f"Expected a VulkanBackend delegate (et_vk.sdpa {cfg.name})"
                )

    def test_golden_matches_eager(self) -> None:
        # The custom op's eager math must equal F.scaled_dot_product_attention,
        # so the on-device golden (computed from the op) is not self-fulfilling.
        for cfg in CONFIGS:
            with self.subTest(config=cfg.name):
                q, k, v = _qkv(cfg)
                mask = _mask(cfg)
                got = torch.ops.et_vk.sdpa.default(q, k, v, mask, None)
                ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-3)
