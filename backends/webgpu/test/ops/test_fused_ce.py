# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused cross-entropy training op (`et_vk.fused_ce`) export + fp64 golden.

`fused_ce(logits[M,V], labels[M], n_valid) -> (loss, dlogits[M,V])` computes the
mean-over-valid CE loss and its gradient in one op (labels < 0 are ignored/pad).
Golden is the fp64 reference (`logsumexp - picked`, `softmax - onehot`), the same math
torch's cross_entropy uses; the native test reconstructs the deterministic inputs.
"""

import os
import unittest
from dataclasses import dataclass

import numpy as np
import torch

from executorch.backends.vulkan import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


@dataclass(frozen=True)
class CeConfig:
    name: str
    m: int  # rows (valid + pad positions)
    v: int  # vocab
    n_pad: int = 0  # trailing rows with label = -1 (ignored)


# Mirrored by the C++ kFusedCeConfigs table. Llama-3.2-1B vocab = 128256.
CONFIGS = [
    CeConfig("tiny", 4, 32),
    CeConfig("masked", 8, 128, n_pad=3),  # some ignored labels
    CeConfig("llama_vocab", 16, 128256),  # real vocab width
]


def _inputs(cfg: CeConfig):
    """Deterministic logits [M,V] + labels [M] (last n_pad = -1); reconstructable in C++."""
    flat = np.arange(cfg.m * cfg.v, dtype=np.int64)
    logits = torch.from_numpy(
        (((flat % 23) - 11).astype(np.float32) / np.float32(8.0)).reshape(cfg.m, cfg.v)
    )
    labels = torch.from_numpy((np.arange(cfg.m, dtype=np.int64) * 7 + 3) % cfg.v)
    if cfg.n_pad:
        labels[cfg.m - cfg.n_pad :] = -1
    n_valid = float(max(1, cfg.m - cfg.n_pad))
    return logits, labels, n_valid


def _fp64_golden(logits: torch.Tensor, labels: torch.Tensor, n_valid: float):
    mask = labels >= 0
    safe = labels.clamp(min=0).long()
    lg = logits.double()
    lse = torch.logsumexp(lg, dim=-1)
    picked = lg.gather(-1, safe[:, None]).squeeze(-1)
    loss = torch.where(mask, (lse - picked) / n_valid, torch.zeros_like(lse)).sum()
    softmax = torch.softmax(lg, dim=-1)
    onehot = torch.nn.functional.one_hot(safe, logits.shape[-1]).double()
    dlogits = torch.where(
        mask[:, None], (softmax - onehot) / n_valid, torch.zeros_like(softmax)
    )
    return loss.to(torch.float32), dlogits.to(torch.float32)


class _CeModule(torch.nn.Module):
    def forward(self, logits, labels, n_valid):
        return torch.ops.et_vk.fused_ce(logits, labels, n_valid)


def _export(logits, labels, n_valid):
    ep = torch.export.export(_CeModule(), (logits, labels, n_valid))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


class TestFusedCe(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for cfg in CONFIGS:
            if cfg.v > 1024:  # width-independent; skip the 128k fixture
                continue
            with self.subTest(config=cfg.name):
                logits, labels, n_valid = _inputs(cfg)
                et = _export(logits, labels, n_valid)
                found = any(
                    d.id == "VulkanBackend"
                    for plan in et.executorch_program.execution_plan
                    for d in plan.delegates
                )
                self.assertTrue(found, f"no VulkanBackend delegate in {cfg.name}")

    def test_op_matches_fp64_golden(self) -> None:
        for cfg in CONFIGS:
            if cfg.v > 1024:
                continue
            with self.subTest(config=cfg.name):
                logits, labels, n_valid = _inputs(cfg)
                loss, dlogits = torch.ops.et_vk.fused_ce(logits, labels, n_valid)
                g_loss, g_dlogits = _fp64_golden(logits, labels, n_valid)
                torch.testing.assert_close(loss, g_loss, atol=5e-4, rtol=1e-3)
                torch.testing.assert_close(dlogits, g_dlogits, atol=5e-4, rtol=1e-3)


def export_fused_ce_model(cfg: CeConfig, pte_path: str, golden_path: str) -> None:
    logits, labels, n_valid = _inputs(cfg)
    et = _export(logits, labels, n_valid)
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    g_loss, g_dlogits = _fp64_golden(logits, labels, n_valid)
    # loss scalar then dlogits, both raw LE fp32
    np.concatenate(
        [g_loss.reshape(1).numpy(), g_dlogits.reshape(-1).numpy()]
    ).astype("<f4").tofile(golden_path)
    print(f"Exported {pte_path}; golden {golden_path}")


def export_all_fused_ce_models(out_dir: str) -> None:
    for cfg in CONFIGS:
        pte = os.path.join(out_dir, f"fused_ce_{cfg.name}.pte")
        golden = os.path.join(out_dir, f"fused_ce_{cfg.name}.golden.bin")
        export_fused_ce_model(cfg, pte, golden)


if __name__ == "__main__":
    unittest.main()
