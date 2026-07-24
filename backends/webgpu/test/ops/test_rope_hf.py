# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HuggingFace rotate-half rotary positional embedding
(`et_vk.apply_rotary_emb_hf`) export + goldens for the WebGPU backend.

Qwen3 (and other HF-derived models) use the rotate-half RoPE convention, which
fuses under VulkanPartitioner into `et_vk.apply_rotary_emb_hf.default` (a full
[max_seq, rotary_dim] freqs table + a start_pos offset, two outputs xq_out,
xk_out as a ValueList). This is the counterpart of test_rope.py for the
interleaved (Llama) convention. Full rotary only (rotary_dim == head_dim), which
is what Qwen3 uses; the runtime rejects partial rotary.

Inputs are deterministic /16 ramps so the native binary reconstructs them
bit-for-bit; the two torch-computed goldens are written for the native binary to
compare (it has no ATen).
"""

import unittest
from collections import namedtuple

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401

import torch
from executorch.backends.vulkan import VulkanPartitioner
from executorch.examples.models.llama.rope import hf_apply_rotary_emb
from executorch.exir import to_edge_transform_and_lower

# B batch, S tokens, NH query heads, NKV kv heads (NH != NKV so the two outputs
# are distinguishable by numel), HD head dim (even; full rotary, rotary_dim==HD).
Shape = namedtuple("Shape", ["name", "b", "s", "nh", "nkv", "hd"])
SHAPES = [
    Shape("multi", 1, 5, 8, 2, 64),
    # Single-token decode at a Qwen3-0.6B-like head config (GQA 16:8, head_dim
    # 128) so the seq=1 / batch decompositions are covered at decode too.
    Shape("decode", 1, 1, 16, 8, 128),
]


class HfRope(torch.nn.Module):
    # unsqueeze_dim=1: freqs [S, HD] -> [S, 1, HD] broadcasts over (B, NH) of the
    # [B, S, NH, HD] q/k, matching the WebGPU kernel + HfRotaryEmbeddingPattern.
    def forward(self, xq, xk, freqs_cos, freqs_sin):
        return hf_apply_rotary_emb(xq, xk, freqs_cos, freqs_sin, unsqueeze_dim=1)


def _ramp(numel: int, mod: int, off: int) -> torch.Tensor:
    # ((i % mod) - off) / 16: exact in fp32, matches test_webgpu_native.cpp.
    idx = torch.arange(numel, dtype=torch.int64)
    return ((idx % mod) - off).to(torch.float32) / 16.0


def _inputs(
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xq = _ramp(shape.b * shape.s * shape.nh * shape.hd, 17, 8).reshape(
        shape.b, shape.s, shape.nh, shape.hd
    )
    xk = _ramp(shape.b * shape.s * shape.nkv * shape.hd, 13, 6).reshape(
        shape.b, shape.s, shape.nkv, shape.hd
    )
    # HF freqs are the FULL rotary_dim (== head_dim) table, not head_dim/2.
    freqs_cos = _ramp(shape.s * shape.hd, 11, 5).reshape(shape.s, shape.hd)
    freqs_sin = _ramp(shape.s * shape.hd, 7, 3).reshape(shape.s, shape.hd)
    return xq, xk, freqs_cos, freqs_sin


def _golden(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Reference = the registered et_vk op the kernel implements (start_pos=0).
    return torch.ops.et_vk.apply_rotary_emb_hf.default(xq, xk, freqs_cos, freqs_sin, 0)


def _export(inputs):
    ep = torch.export.export(HfRope().eval(), inputs)
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


class TestRopeHf(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for shape in SHAPES:
            with self.subTest(shape=shape.name):
                et = _export(_inputs(shape))
                found = any(
                    d.id == "VulkanBackend"
                    for plan in et.executorch_program.execution_plan
                    for d in plan.delegates
                )
                self.assertTrue(
                    found,
                    "Expected a VulkanBackend delegate (apply_rotary_emb_hf " "fusion)",
                )

    def test_golden_matches_eager(self) -> None:
        # The et_vk golden must equal the real HF rotate-half apply_rotary_emb,
        # so a buggy golden can't fake-pass the native kernel. Run at both shapes
        # so the S=1 decode position indexing is covered.
        for shape in SHAPES:
            with self.subTest(shape=shape.name):
                xq, xk, fc, fs = _inputs(shape)
                gq, gk = _golden(xq, xk, fc, fs)
                eq, ek = hf_apply_rotary_emb(xq, xk, fc, fs, unsqueeze_dim=1)
                torch.testing.assert_close(gq, eq, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(gk, ek, atol=1e-5, rtol=1e-5)


def export_rope_hf_model(
    pte_path: str, xq_golden_path: str, xk_golden_path: str, shape_name: str = "multi"
) -> None:
    """Write the apply_rotary_emb_hf .pte + the xq_out and xk_out torch goldens
    (raw LE fp32). Inputs are /16 ramps reconstructed in the native test."""
    shape = next(s for s in SHAPES if s.name == shape_name)
    xq, xk, fc, fs = _inputs(shape)
    gq, gk = _golden(xq, xk, fc, fs)
    et = _export((xq, xk, fc, fs))
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    gq.detach().numpy().astype("<f4").tofile(xq_golden_path)
    gk.detach().numpy().astype("<f4").tofile(xk_golden_path)
    print(
        f"Exported {pte_path} (shape={shape_name}); xq_out golden {xq_golden_path} "
        f"({gq.numel()} floats); xk_out golden {xk_golden_path} ({gk.numel()} floats)"
    )


if __name__ == "__main__":
    unittest.main()
