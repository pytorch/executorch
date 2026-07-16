# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fp32 update_cache (KV-cache write) export tests via VulkanPartitioner.

Verifies the export/delegation side here; on-GPU numerics are checked by the
dedicated native test `test/native/test_update_cache.cpp`: single-shot cases
(non-zero input_pos + varied shapes) via `export_update_cache_cases`, and the
multi-step advancing-input_pos replay (mirroring VulkanSDPATest) via
`export_update_cache_replay`. update_cache scatters a projected value tensor
[1, S, H, D] into the KV cache [1, Cmax, H, D] at the sequence offset input_pos.
"""

import os
import unittest

import torch

# Importing custom_ops registers torch.ops.llama.update_cache (the schema lives
# in the C++ AOT lib loaded here).
from executorch.backends.vulkan import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401


class UpdateCacheModule(torch.nn.Module):
    """Writes the projected value into the KV cache at input_pos."""

    def __init__(self, input_pos: int = 0) -> None:
        super().__init__()
        self.input_pos = input_pos

    def forward(self, value: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        return torch.ops.llama.update_cache(value, cache, self.input_pos)


class TestUpdateCache(unittest.TestCase):
    def _export_and_check(self, model, example_inputs) -> None:
        ep = torch.export.export(model, example_inputs)
        et_program = to_edge_transform_and_lower(
            ep, partitioner=[VulkanPartitioner()]
        ).to_executorch()

        found_vulkan = False
        for plan in et_program.executorch_program.execution_plan:
            for delegate in plan.delegates:
                if delegate.id == "VulkanBackend":
                    found_vulkan = True
                    break
        self.assertTrue(found_vulkan, "Expected VulkanBackend delegate in .pte")

    def test_update_cache_prefill_small(self) -> None:
        # input_pos=0 prefill: value [1,S=2,H=2,D=4] into cache [1,Cmax=8,H=2,D=4].
        value = torch.randn(1, 2, 2, 4)
        cache = torch.zeros(1, 8, 2, 4)
        self._export_and_check(UpdateCacheModule(0), (value, cache))

    def test_update_cache_gqa_shapes(self) -> None:
        # GQA-style: fewer kv heads, larger head dim.
        value = torch.randn(1, 3, 2, 8)
        cache = torch.zeros(1, 16, 2, 8)
        self._export_and_check(UpdateCacheModule(0), (value, cache))


def export_update_cache_model(output_path: str) -> None:
    """Export an update_cache model to .pte for the native runtime test.

    Shapes match the native test: value [1,S=2,H=2,D=4] into cache
    [1,Cmax=8,H=2,D=4] at input_pos=0. Example tensor *values* here are only for
    tracing; the native test supplies its own deterministic inputs at runtime.
    """
    S, H, D, Cmax = 2, 2, 4, 8
    model = UpdateCacheModule(0)
    value = torch.zeros(1, S, H, D)
    cache = torch.zeros(1, Cmax, H, D)
    ep = torch.export.export(model, (value, cache))
    et_program = to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)
    print(f"Exported {output_path}")


# (name, S, H, D, Cmax, input_pos) -- mirrors kCases in
# test/native/test_update_cache.cpp. Covers non-zero input_pos (the dst_offset
# path) and a second head_dim/n_heads shape. All writes stay in-bounds.
_NATIVE_CASES = [
    ("prefill", 2, 2, 4, 8, 0),
    ("offset", 2, 2, 4, 8, 5),
    ("shape_b", 3, 4, 8, 16, 0),
    ("shape_b_offset", 3, 4, 8, 16, 10),
]


def export_update_cache_cases(out_dir: str) -> None:
    """Export one .pte per native test case (input_pos baked).

    The native test supplies deterministic inputs and computes the integer-exact
    scatter reference inline, so only the .pte (shapes + input_pos baked) is
    written here -- no golden file.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name, s, h, d, cmax, input_pos in _NATIVE_CASES:
        model = UpdateCacheModule(input_pos)
        value = torch.zeros(1, s, h, d)
        cache = torch.zeros(1, cmax, h, d)
        ep = torch.export.export(model, (value, cache))
        et_program = to_edge_transform_and_lower(
            ep, partitioner=[VulkanPartitioner()]
        ).to_executorch()
        with open(os.path.join(out_dir, f"{name}.pte"), "wb") as f:
            f.write(et_program.buffer)
        print(f"Exported {name}.pte (input_pos={input_pos})")


# (name, num_kv_heads, head_dim, seq_lens) -- mirrors the VulkanSDPATest param
# sets (sdpa_test.cpp:855-881). Cmax = sum(seq_lens) (exact fit). The native test
# threads the returned cache across steps as input_pos advances by seq_len.
_REPLAY_SEQS = [
    ("seqA", 4, 4, [3, 1, 1, 5, 1, 1, 2]),
    ("seqB", 2, 8, [3, 1, 1, 5, 1, 1]),
    ("llama3", 8, 128, [111, 1, 1, 1, 57, 1, 1]),
]


def export_update_cache_replay(out_dir: str) -> None:
    """Export one .pte per replay step (seq_len + input_pos baked).

    Mirrors Vulkan's multi-step advancing-input_pos cache accumulation; the
    native test feeds the returned cache into the next step and checks the
    integer-exact scatter golden after each write -- no golden file.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name, h, d, seqs in _REPLAY_SEQS:
        cmax = sum(seqs)
        input_pos = 0
        for idx, s in enumerate(seqs):
            model = UpdateCacheModule(input_pos)
            value = torch.zeros(1, s, h, d)
            cache = torch.zeros(1, cmax, h, d)
            ep = torch.export.export(model, (value, cache))
            et_program = to_edge_transform_and_lower(
                ep, partitioner=[VulkanPartitioner()]
            ).to_executorch()
            fname = f"{name}_step{idx}_S{s}_pos{input_pos}.pte"
            with open(os.path.join(out_dir, fname), "wb") as f:
                f.write(et_program.buffer)
            print(f"Exported {fname}")
            input_pos += s


# (name, value_shape, cache_shape, dtype) -- each violates one runtime guard but
# still delegates to VulkanBackend at export (ATen's update_cache meta allows
# it). The WebGPU backend must reject each at graph build; the native test
# asserts a graceful delegate error (no crash, no silent-wrong output). The
# other guards (head_dim/n_heads mismatch, non-4D, out-of-bounds start_pos) are
# rejected by ATen at export, so they cannot be baked into a .pte.
_NEGATIVE_CASES = [
    ("neg_batch", (2, 2, 2, 4), (2, 8, 2, 4), torch.float32),  # batch must be 1
    ("neg_fp16", (1, 2, 2, 4), (1, 8, 2, 4), torch.float16),  # fp32-only
]


def export_update_cache_negative(out_dir: str) -> None:
    """Export guard-violating .pte's the WebGPU backend must reject at build.

    Asserts each still delegates to VulkanBackend, so the native test exercises
    the runtime guard rather than a CPU-fallback path.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name, vshape, cshape, dtype in _NEGATIVE_CASES:
        model = UpdateCacheModule(0)
        value = torch.zeros(*vshape, dtype=dtype)
        cache = torch.zeros(*cshape, dtype=dtype)
        ep = torch.export.export(model, (value, cache))
        et_program = to_edge_transform_and_lower(
            ep, partitioner=[VulkanPartitioner()]
        ).to_executorch()
        delegated = any(
            d.id == "VulkanBackend"
            for plan in et_program.executorch_program.execution_plan
            for d in plan.delegates
        )
        if not delegated:
            raise RuntimeError(f"{name}: expected VulkanBackend delegation")
        with open(os.path.join(out_dir, f"{name}.pte"), "wb") as f:
            f.write(et_program.buffer)
        print(f"Exported {name}.pte")


if __name__ == "__main__":
    unittest.main()
