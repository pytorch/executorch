# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""4-bit groupwise-symmetric quantized embedding (`et_vk.embedding_q4gsw`) export
+ golden for the WebGPU backend.

Quantizes an nn.Embedding with the Llama EmbeddingQuantHandler recipe (int4
groupwise-symmetric, packed) which lowers to `quantized_decomposed.embedding_4bit`
and fuses under VulkanPartitioner into `et_vk.embedding_q4gsw.default`
(is_linear_weight=False). Writes a torch-computed golden (the native binary has no
ATen) via the registered et_vk reference op + the raw int32 indices the native
test loads and compares.

Two shapes are exercised: a tiny one and a Llama-3.2-1B-scale one (EMBED=2048,
GROUP=64) so the per-group scale indexing (32 groups/row) + dequant are validated
at the real embedding dim, not just a single 64-wide row.
"""

import unittest
from collections import namedtuple

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401

import torch
from executorch.backends.vulkan import VulkanPartitioner
from executorch.examples.models.llama.source_transformation.quantize import (
    EmbeddingQuantHandler,
)
from executorch.exir import to_edge_transform_and_lower

# vocab rows, embed columns (embed % 32 == 0), group-wise scales, gather indices.
Shape = namedtuple("Shape", ["name", "vocab", "embed", "group", "indices"])
SHAPES = [
    Shape("small", 64, 64, 32, [1, 5, 63, 0]),
    # Llama-3.2-1B embedding dim + group (small vocab keeps the export light).
    Shape("llama1b", 512, 2048, 64, [1, 5, 511, 0]),
]


class _EmbeddingModel(torch.nn.Module):
    def __init__(self, vocab: int, embed: int) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, embed)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.emb(idx)


def _make_quantized_model(shape: Shape) -> torch.nn.Module:
    torch.manual_seed(0)
    return (
        EmbeddingQuantHandler(
            _EmbeddingModel(shape.vocab, shape.embed).eval(),
            device="cpu",
            bitwidth=4,
            group_size=shape.group,
            packed=True,
            quantize_with_hqq=False,
        )
        .quantized_model()
        .eval()
    )


def _indices(shape: Shape) -> torch.Tensor:
    return torch.tensor(shape.indices, dtype=torch.long)


def _quant_params(qm: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, int]:
    sd = qm.state_dict()
    weight = next(
        v for k, v in sd.items() if k.endswith("weight") and v.dtype == torch.uint8
    )
    scales = next(v for k, v in sd.items() if k.endswith("scales"))
    if scales.ndim == 1:
        scales = scales.unsqueeze(1)
    embed = weight.shape[1] * 2
    group_size = embed // scales.shape[1]
    return weight, scales, group_size


def _golden(qm: torch.nn.Module, idx: torch.Tensor) -> torch.Tensor:
    # Reference = the registered et_vk dequant+gather op (non-linear branch).
    weight, scales, group_size = _quant_params(qm)
    return torch.ops.et_vk.embedding_q4gsw.default(
        weight, scales, group_size, idx, False
    )


def _export(qm: torch.nn.Module, idx: torch.Tensor):
    ep = torch.export.export(qm, (idx,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


class TestEmbeddingQ4gsw(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for shape in SHAPES:
            with self.subTest(shape=shape.name):
                et = _export(_make_quantized_model(shape), _indices(shape))
                found = any(
                    d.id == "VulkanBackend"
                    for plan in et.executorch_program.execution_plan
                    for d in plan.delegates
                )
                self.assertTrue(
                    found, "Expected a VulkanBackend delegate (embedding_q4gsw fusion)"
                )

    def test_golden_matches_eager(self) -> None:
        # The torch golden (et_vk reference) must equal torch dequant+gather, so a
        # buggy golden can't fake-pass the native kernel. Run at both shapes so the
        # Llama-scale per-group scale indexing (32 groups/row) is covered.
        for shape in SHAPES:
            with self.subTest(shape=shape.name):
                qm = _make_quantized_model(shape)
                idx = _indices(shape)
                weight, scales, group_size = _quant_params(qm)
                vocab = weight.shape[0]
                embed = weight.shape[1] * 2
                # fp64 reference dequant, vectorized (no fp32 rounding in oracle).
                w = weight.to(torch.int64)
                nib = torch.empty((vocab, embed), dtype=torch.int64)
                nib[:, 0::2] = (w >> 4) & 0xF  # even dim -> high nibble
                nib[:, 1::2] = w & 0xF  # odd dim -> low nibble
                scale_exp = scales.to(torch.float64).repeat_interleave(
                    group_size, dim=1
                )
                deq = (nib - 8).to(torch.float64) * scale_exp
                eager = torch.nn.functional.embedding(idx, deq)
                golden = _golden(qm, idx)
                torch.testing.assert_close(golden.double(), eager, atol=1e-5, rtol=1e-5)


def export_embedding_q4gsw_model(
    pte_path: str, golden_path: str, indices_path: str, shape_name: str = "small"
) -> None:
    """Write the embedding_q4gsw .pte + torch golden (raw LE fp32) + raw LE int32
    indices (downcast from int64 for the int32-typed GPU buffer). `shape_name`
    selects an entry from SHAPES (default the tiny shape; "llama1b" = EMBED=2048)."""
    shape = next(s for s in SHAPES if s.name == shape_name)
    qm = _make_quantized_model(shape)
    idx = _indices(shape)
    golden = _golden(qm, idx).detach().numpy().astype("<f4")
    et = _export(qm, idx)
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    golden.tofile(golden_path)
    idx.to(torch.int32).numpy().astype("<i4").tofile(indices_path)
    print(
        f"Exported {pte_path} (shape={shape_name}); golden {golden_path} "
        f"({golden.size} floats); indices {indices_path} ({idx.numel()} int32)"
    )


if __name__ == "__main__":
    unittest.main()
