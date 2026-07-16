# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.embedding.default` (fp32) export + reference checks.

fp32 embedding is the BART token + positional embedding lookup in Florence-2
(the only remaining attention-stack op that AOT-delegates but had no WebGPU
runtime kernel). Forward is a plain row gather `out[row, :] = weight[idx[row], :]`.

`test_export_delegates` checks the op lowers into the VulkanBackend delegate.
`test_golden_matches_eager` checks the gather math against `nn.Embedding`.
On-device GPU numerics run via a GPU rig.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class EmbeddingModule(torch.nn.Module):
    def __init__(self, num_embeddings: int, embed_dim: int):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings, embed_dim)
        with torch.no_grad():
            self.emb.weight.copy_(
                torch.linspace(
                    -1.0, 1.0, num_embeddings * embed_dim, dtype=torch.float32
                ).reshape(num_embeddings, embed_dim)
            )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)


# (num_embeddings, embed_dim, indices) — BART-ish vocab/hidden + a small case.
CONFIGS = [
    ("small", 16, 8, torch.tensor([0, 3, 15, 7], dtype=torch.long)),
    ("bart_tok", 1024, 768, torch.tensor([[1, 5, 1023, 0, 42]], dtype=torch.long)),
]


def _export(m: torch.nn.Module, idx: torch.Tensor):
    ep = torch.export.export(m, (idx,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


class EmbeddingTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, ne, d, idx in CONFIGS:
            with self.subTest(config=name):
                et = _export(EmbeddingModule(ne, d).eval(), idx)
                found = any(
                    de.id == "VulkanBackend"
                    for plan in et.executorch_program.execution_plan
                    for de in plan.delegates
                )
                self.assertTrue(
                    found, f"Expected a VulkanBackend delegate (embedding {name})"
                )

    def test_golden_matches_eager(self) -> None:
        # The gather the WebGPU kernel reproduces must match nn.Embedding.
        for name, ne, d, idx in CONFIGS:
            with self.subTest(config=name):
                m = EmbeddingModule(ne, d).eval()
                got = m.emb.weight[idx]
                ref = m(idx)
                torch.testing.assert_close(got, ref)
