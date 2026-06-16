# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Constant-tensor prepack (`et_vk.prepack`) export + golden for the WebGPU
backend.

The VulkanPartitioner wraps every constant feeding a delegated op in an
`et_vk.prepack.default` node that materializes the constant into a GPU buffer at
init. Model `M(x) = x + w` (w a constant) routes `w` through prepack, so the
delegate must run the prepack copy for the output to equal `x + w` rather than
`x + 0 = x`. The input is a deterministic /16 ramp so the native binary
reconstructs it bit-for-bit; the torch-computed golden is written for the native
binary to compare (it has no ATen).
"""

import unittest

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401

import torch
from executorch.backends.vulkan import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# 4x4 constant weight, small enough to dump and reason about by hand.
N = 4


class _AddConst(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # arange weight: non-zero everywhere so an unrun prepack (out = x + 0 = x)
        # is unambiguously distinguishable from a correct one (out = x + w).
        self.w = torch.nn.Parameter(
            torch.arange(N * N, dtype=torch.float32).reshape(N, N)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w


class _AddTwoConst(torch.nn.Module):
    # Two constants => two prepack nodes (the multi-copy path E2E Llama needs);
    # add-only so it stays delegated with just this stack's registered ops.
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Parameter(
            torch.arange(N * N, dtype=torch.float32).reshape(N, N)
        )
        self.w2 = torch.nn.Parameter(
            torch.arange(N * N, dtype=torch.float32).reshape(N, N) * 0.5 - 3.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w1 + self.w2


class _AddTiedConst(torch.nn.Module):
    # Two BYTE-IDENTICAL constants => two prepack nodes sharing ONE SHA256
    # named-data key (tied/duplicate weights). Exercises the prepack handler
    # materializing the same key twice (independent get_data + Free per call).
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Parameter(
            torch.arange(N * N, dtype=torch.float32).reshape(N, N)
        )
        self.w2 = torch.nn.Parameter(
            torch.arange(N * N, dtype=torch.float32).reshape(N, N)
        )
        # Pin the tied premise; the dedup to one key is assumed, not asserted.
        assert torch.equal(self.w1, self.w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w1 + self.w2


def _inputs() -> tuple[torch.Tensor]:
    # ((i % 13) - 6) / 16: exact in fp32, matches test_webgpu_native.cpp.
    idx = torch.arange(N * N, dtype=torch.int64)
    x = (((idx % 13) - 6).to(torch.float32) / 16.0).reshape(N, N)
    return (x,)


def _export(model, inputs):
    ep = torch.export.export(model.eval(), inputs)
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


class TestPrepack(unittest.TestCase):
    def test_export_delegates(self) -> None:
        # Each model must fully delegate -- every constant wrapped in a prepack
        # node inside a VulkanBackend delegate (single, multi-const, tied).
        for name, model in (
            ("x + w", _AddConst()),
            ("x + w1 + w2", _AddTwoConst()),
            ("x + w + w (tied)", _AddTiedConst()),
        ):
            with self.subTest(model=name):
                et = _export(model, _inputs())
                found = any(
                    d.id == "VulkanBackend"
                    for plan in et.executorch_program.execution_plan
                    for d in plan.delegates
                )
                self.assertTrue(found, f"Expected a VulkanBackend delegate: {name}")


def _write(model, pte_path: str, golden_path: str) -> None:
    (x,) = _inputs()
    golden = model.eval()(x)
    et = _export(model, (x,))
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    golden.detach().numpy().astype("<f4").tofile(golden_path)
    print(f"Exported {pte_path}; golden {golden_path} ({golden.numel()} floats)")


def export_prepack_model(pte_path: str, golden_path: str) -> None:
    """Write the x + w .pte + torch golden (raw LE fp32). One prepacked constant.
    The input is a /16 ramp reconstructed in the native test."""
    _write(_AddConst(), pte_path, golden_path)


def export_prepack_two_const_model(pte_path: str, golden_path: str) -> None:
    """Write the x + w1 + w2 .pte + golden. Two prepacked constants, exercising
    the multi-copy path."""
    _write(_AddTwoConst(), pte_path, golden_path)


def export_prepack_tied_const_model(pte_path: str, golden_path: str) -> None:
    """Write the x + w1 + w2 .pte + golden where w1 and w2 are BYTE-IDENTICAL,
    so they share one named-data key -> two prepack nodes materialize the same
    key (verifies per-call buffer ownership / no double-free on tied weights)."""
    _write(_AddTiedConst(), pte_path, golden_path)


if __name__ == "__main__":
    unittest.main()
