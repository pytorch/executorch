# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten._log_softmax.default` export + golden for the WebGPU backend.

Exports single-op log-softmax graphs through VulkanPartitioner and writes a
torch-computed golden (the native binary has no ATen) + the raw fp32 input the
native test loads and compares. log_softmax is on the training critical path:
the cross-entropy / decomposed-backward lowers to `_log_softmax`, computed in
kernel as `x - (max + log(sum exp(x - max)))`. `dim=-1` gives inner=1; a middle
dim exercises the inner>1 reduction path.
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class LogSoftmaxModule(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(x, dim=self.dim)


def _det_input() -> torch.Tensor:
    """Deterministic fp32 spanning large +/- magnitudes (exercises the
    max-subtraction: a naive exp(x) would overflow on the +40 entries)."""
    return torch.linspace(-40.0, 40.0, 4 * 8 * 16, dtype=torch.float32).reshape(
        4, 8, 16
    )


def _export(m: torch.nn.Module, x: torch.Tensor):
    ep = torch.export.export(m, (x,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegates(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class TestLogSoftmax(unittest.TestCase):
    def test_export_delegates_last_dim(self) -> None:
        et = _export(LogSoftmaxModule(-1).eval(), _det_input())
        self.assertTrue(
            _delegates(et), "Expected a VulkanBackend delegate (log_softmax dim=-1)"
        )

    def test_export_delegates_middle_dim(self) -> None:
        # dim=1 => inner>1: the non-unit-stride reduction path in the kernel.
        et = _export(LogSoftmaxModule(1).eval(), _det_input())
        self.assertTrue(
            _delegates(et), "Expected a VulkanBackend delegate (log_softmax dim=1)"
        )

    def test_golden_matches_eager(self) -> None:
        x = _det_input()
        torch.testing.assert_close(
            LogSoftmaxModule(-1)(x), torch.log_softmax(x, dim=-1)
        )


def export_log_softmax_model(pte_path: str, golden_path: str, input_path: str) -> None:
    """Write log_softmax(dim=-1) .pte + torch golden (raw LE fp32) + raw LE fp32 input."""
    m = LogSoftmaxModule(-1).eval()
    x = _det_input()
    golden = m(x).detach().numpy().astype("<f4")
    et = _export(m, x)
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    golden.tofile(golden_path)
    x.numpy().astype("<f4").tofile(input_path)
    print(
        f"Exported {pte_path}; golden {golden_path} ({golden.size} floats); "
        f"input {input_path} ({x.numel()} floats)"
    )


if __name__ == "__main__":
    unittest.main()
