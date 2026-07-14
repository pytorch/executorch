# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AdamW optimizer step (`et_vk.adamw_step`) export + fp64 golden.

`adamw_step(param, m, v, grad, lr, beta1, beta2, eps, weight_decay, bc1, bc2)`
updates the fp32 latent in place: decoupled weight decay, then the bias-corrected
Adam moment update. `bc1`/`bc2` (= 1 - beta^t) are host-precomputed so the kernel
carries no step counter. The op mutates and returns `param`/`m`/`v` (aliased), so
export wraps it in `auto_functionalized`, which VulkanPartitioner tags by name (the
same mutating-op path as `update_cache`). Golden is the fp64 reference, computed
independently so a lossy fp32 op impl cannot fake-pass.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)
from executorch.exir import to_edge_transform_and_lower

# torch.optim.AdamW defaults; bc1/bc2 are 1 - beta^step (host-precomputed).
LR = 1e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8


@dataclass(frozen=True)
class AdamwConfig:
    name: str
    numel: int
    weight_decay: float = 0.01
    step: int = 1


CONFIGS = [
    AdamwConfig("small", 64),
    AdamwConfig("no_wd", 256, weight_decay=0.0),  # wd=0 -> the plain Adam path
    AdamwConfig("later_step", 1000, step=10),  # bias correction well past t=1
]


def _inputs(cfg: AdamwConfig):
    """Deterministic fp32 param/m/v/grad + the host bias corrections."""
    g = torch.Generator().manual_seed(0)
    param = torch.randn(cfg.numel, generator=g, dtype=torch.float32)
    m = torch.randn(cfg.numel, generator=g, dtype=torch.float32) * 0.1
    v = torch.rand(cfg.numel, generator=g, dtype=torch.float32) * 0.01
    grad = torch.randn(cfg.numel, generator=g, dtype=torch.float32)
    bc1 = 1.0 - BETA1**cfg.step
    bc2 = 1.0 - BETA2**cfg.step
    return param, m, v, grad, bc1, bc2


def _fp64_golden(param, m, v, grad, wd, bc1, bc2):
    """fp64 truth for one AdamW step; mirrors adamw_step.wgsl exactly."""
    p = param.double()
    g = grad.double()
    p = p - LR * wd * p
    m64 = BETA1 * m.double() + (1.0 - BETA1) * g
    v64 = BETA2 * v.double() + (1.0 - BETA2) * g * g
    mhat = m64 / bc1
    vhat = v64 / bc2
    p = p - LR * mhat / (torch.sqrt(vhat) + EPS)
    return p.float(), m64.float(), v64.float()


class _AdamwModule(torch.nn.Module):
    def __init__(self, wd: float, bc1: float, bc2: float) -> None:
        super().__init__()
        self.wd = wd
        self.bc1 = bc1
        self.bc2 = bc2

    def forward(self, param, m, v, grad):
        return torch.ops.et_vk.adamw_step(
            param, m, v, grad, LR, BETA1, BETA2, EPS, self.wd, self.bc1, self.bc2
        )


def _export(cfg: AdamwConfig):
    param, m, v, grad, bc1, bc2 = _inputs(cfg)
    ep = torch.export.export(
        _AdamwModule(cfg.weight_decay, bc1, bc2), (param, m, v, grad)
    )
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegates(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class TestAdamwStep(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for cfg in CONFIGS:
            with self.subTest(config=cfg.name):
                et = _export(cfg)
                self.assertTrue(
                    _delegates(et), f"no VulkanBackend delegate in {cfg.name}"
                )

    def test_op_matches_fp64_golden(self) -> None:
        for cfg in CONFIGS:
            with self.subTest(config=cfg.name):
                param, m, v, grad, bc1, bc2 = _inputs(cfg)
                g_param, g_m, g_v = _fp64_golden(
                    param, m, v, grad, cfg.weight_decay, bc1, bc2
                )
                # Op mutates in place; clone so the golden saw the originals.
                out_param, out_m, out_v = torch.ops.et_vk.adamw_step(
                    param.clone(),
                    m.clone(),
                    v.clone(),
                    grad,
                    LR,
                    BETA1,
                    BETA2,
                    EPS,
                    cfg.weight_decay,
                    bc1,
                    bc2,
                )
                torch.testing.assert_close(out_param, g_param, atol=5e-4, rtol=1e-3)
                torch.testing.assert_close(out_m, g_m, atol=5e-4, rtol=1e-3)
                torch.testing.assert_close(out_v, g_v, atol=5e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
