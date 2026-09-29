# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.embedding.default` (fp32 row-gather) export + golden for the WebGPU backend.

Exports single-op embedding graphs through VulkanPartitioner and checks a torch
golden. embedding is a pure row-gather -- out[i, :] = weight[idx[i], :] -- on the
token-embedding path that feeds the transformer (and the fine-tuning training
window). 1D indices exercise the [S, D] output; 2D indices the batched [B, S, D]
path. The indices span the full vocab (incl. the first/last rows) so a wrong
row-stride would miss the golden.
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (num_embeddings, embedding_dim, indices_shape).
CONFIGS = {
    "rows_1d": (32, 16, (8,)),
    "batched_2d": (128, 8, (2, 4)),
}


class EmbeddingModule(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(idx, self.weight)


def _det_weight(num_embeddings: int, dim: int) -> torch.Tensor:
    """Deterministic fp32 [num_embeddings, dim] table (distinct per-row values)."""
    return torch.linspace(-1.0, 1.0, num_embeddings * dim, dtype=torch.float32).reshape(
        num_embeddings, dim
    )


def _det_indices(num_embeddings: int, shape: tuple[int, ...]) -> torch.Tensor:
    """Deterministic int64 indices spread across the vocab, forced to hit row 0
    and the last row so an off-by-one row-stride shows up in the golden."""
    n = 1
    for s in shape:
        n *= s
    flat = (torch.arange(n, dtype=torch.int64) * 7 + 3) % num_embeddings
    flat[0] = 0
    flat[-1] = num_embeddings - 1
    return flat.reshape(shape)


def _export(m: torch.nn.Module, idx: torch.Tensor):
    ep = torch.export.export(m, (idx,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegates(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class TestEmbedding(unittest.TestCase):
    def test_export_delegates(self) -> None:
        # aten.embedding must be absorbed into the VulkanBackend delegate.
        for name, (num_embeddings, dim, shape) in CONFIGS.items():
            with self.subTest(name=name):
                weight = _det_weight(num_embeddings, dim)
                idx = _det_indices(num_embeddings, shape)
                et = _export(EmbeddingModule(weight).eval(), idx)
                self.assertTrue(
                    _delegates(et),
                    f"Expected a VulkanBackend delegate (embedding {name})",
                )

    def test_golden_matches_eager(self) -> None:
        # fp64 gather golden: out[i,:] == weight[idx[i],:], bit-exact.
        for name, (num_embeddings, dim, shape) in CONFIGS.items():
            with self.subTest(name=name):
                weight = _det_weight(num_embeddings, dim)
                idx = _det_indices(num_embeddings, shape)
                got = EmbeddingModule(weight)(idx)
                golden = torch.nn.functional.embedding(idx, weight.double()).to(
                    torch.float32
                )
                torch.testing.assert_close(got, golden)


if __name__ == "__main__":
    unittest.main()
