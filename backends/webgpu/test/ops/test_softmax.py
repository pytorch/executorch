# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten._softmax.default` export + golden for the WebGPU backend.

Exports single-op softmax graphs through VulkanPartitioner and writes a
torch-computed golden (the native binary has no ATen) + the raw fp32 input the
native test loads and compares. Softmax is on the training critical path: the
decomposed attention lowers to `matmul -> softmax(dim=-1) -> matmul`, whereas the
fused inference `sdpa` computes softmax internally, so a standalone `_softmax`
op is only exercised by the decomposed backward. `dim=-1` gives inner=1 (the
attention case); a middle dim exercises the inner>1 reduction path.
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (shape, dim). dim=-1 => inner=1 (attention); a middle dim => inner>1.
CONFIGS = {
    "last_dim_3d": ((4, 8, 16), -1),
    "middle_dim_3d": ((4, 8, 16), 1),
    "last_dim_2d": ((32, 64), -1),
}


class SoftmaxModule(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=self.dim)


def _det_input(shape) -> torch.Tensor:
    """Deterministic fp32 spanning large +/- magnitudes (exercises the
    max-subtraction: a naive exp(x) would overflow on the +40 entries)."""
    numel = 1
    for d in shape:
        numel *= d
    return torch.linspace(-40.0, 40.0, numel, dtype=torch.float32).reshape(shape)


def _fp64_golden(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically-stable softmax in fp64, independent of torch.softmax:
    exp(x - max) / sum(exp(x - max)) along `dim`."""
    xd = x.double()
    e = torch.exp(xd - xd.amax(dim=dim, keepdim=True))
    return (e / e.sum(dim=dim, keepdim=True)).to(torch.float32)


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


class TestSoftmax(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (shape, dim) in CONFIGS.items():
            with self.subTest(config=name):
                et = _export(SoftmaxModule(dim).eval(), _det_input(shape))
                self.assertTrue(
                    _delegates(et),
                    f"Expected a VulkanBackend delegate (softmax {name})",
                )

    def test_golden_matches_fp64(self) -> None:
        for name, (shape, dim) in CONFIGS.items():
            with self.subTest(config=name):
                x = _det_input(shape)
                torch.testing.assert_close(
                    torch.softmax(x, dim=dim),
                    _fp64_golden(x, dim),
                    atol=1e-6,
                    rtol=1e-5,
                )


def export_softmax_model(pte_path: str, golden_path: str, input_path: str) -> None:
    """Write the softmax(dim=-1) .pte + fp64 golden (raw LE fp32) + raw LE fp32 input."""
    shape = (4, 8, 16)
    m = SoftmaxModule(-1).eval()
    x = _det_input(shape)
    golden = _fp64_golden(x, -1).numpy().astype("<f4")
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
