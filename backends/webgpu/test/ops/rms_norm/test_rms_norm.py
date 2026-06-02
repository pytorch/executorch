# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fp32 RMSNorm export tests via VulkanPartitioner.

Verifies the export side only; numerics are checked in the native test
`test/test_webgpu_native.cpp`.
"""

import os
import unittest

import torch
from executorch.backends.vulkan import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class RmsNormModule(torch.nn.Module):
    """Standard RMSNorm with learnable per-feature weight."""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.to(torch.float32)
        var = x_f32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(var + self.eps)
        return (x_norm * self.weight).to(x.dtype)


class TestRmsNorm(unittest.TestCase):
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
        self.assertGreater(len(et_program.buffer), 100)

    def test_rms_norm_basic_small(self) -> None:
        self._export_and_check(RmsNormModule(64), (torch.randn(1, 1, 1, 64),))

    def test_rms_norm_llm_hidden(self) -> None:
        # LLM-typical hidden size.
        self._export_and_check(RmsNormModule(896), (torch.randn(1, 1, 1, 896),))

    def test_rms_norm_multi_row(self) -> None:
        # Multiple rows along the seq-len dimension (prefill-style).
        self._export_and_check(RmsNormModule(896), (torch.randn(1, 1, 7, 896),))

    def test_rms_norm_4d(self) -> None:
        # 4D shape similar to QK norm with multiple Z slices.
        self._export_and_check(RmsNormModule(128), (torch.randn(1, 5, 4, 128),))


def export_rms_norm_model(output_path: str) -> None:
    """Export the RMSNorm model to .pte for the native runtime test."""
    hidden = 896
    seq_len = 7
    model = RmsNormModule(hidden, eps=1e-6)
    # Fix the weight to a known value the native test reconstructs.
    with torch.no_grad():
        model.weight.copy_(torch.linspace(0.5, 1.5, hidden, dtype=torch.float32))
    example_inputs = (torch.randn(1, 1, seq_len, hidden),)
    ep = torch.export.export(model, example_inputs)
    et_program = to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)
    print(f"Exported {output_path}")


def _ramp(shape) -> torch.Tensor:
    """Deterministic linear ramp in [-1, 1] reshaped to `shape`."""
    n = 1
    for d in shape:
        n *= d
    return torch.linspace(-1.0, 1.0, n, dtype=torch.float32).reshape(shape)


def _linspace_weight(hidden: int) -> torch.Tensor:
    return torch.linspace(0.5, 1.5, hidden, dtype=torch.float32)


def _distinct_rows(shape) -> torch.Tensor:
    """Each row is a ramp scaled by 10^(r-2) so rows differ sharply in magnitude."""
    rows, width = shape[-2], shape[-1]
    base = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    stacked = torch.stack([base * (10.0 ** (r - 2)) for r in range(rows)])
    return stacked.reshape(shape)


def _mixed_sign(shape) -> torch.Tensor:
    """Row 0 all-negative, row 1 near-zero (eps-dominated), row 2 mixed, row 3 positive."""
    width = shape[-1]
    base = torch.linspace(0.1, 1.0, width, dtype=torch.float32)
    sign = torch.tensor([1.0, -1.0], dtype=torch.float32).repeat(width // 2)
    stacked = torch.stack(
        [-base, torch.full((width,), 1e-4, dtype=torch.float32), base * sign, base]
    )
    return stacked.reshape(shape)


def _weight_zeros_neg(hidden: int) -> torch.Tensor:
    """Spans negatives to positives with forced zeros (no weight>0 assumption)."""
    w = torch.linspace(-1.0, 1.0, hidden, dtype=torch.float32).clone()
    w[0] = 0.0
    w[hidden // 2] = 0.0
    return w


# Coverage cases (ssjia, D106887028): each bakes weight+shape -> own .pte; eps=1e-6.
_CASES = [
    {"name": "baseline", "shape": (1, 1, 7, 896)},
    {"name": "width_eq_wg", "shape": (1, 1, 1, 64)},
    {"name": "width_lt_wg", "shape": (1, 1, 1, 32)},
    {
        "name": "width_1",
        "shape": (1, 1, 1, 1),
        "weight_fn": lambda h: torch.tensor([1.3], dtype=torch.float32),
        "input_fn": lambda s: torch.tensor([0.7], dtype=torch.float32).reshape(s),
    },
    {"name": "width_100", "shape": (1, 1, 1, 100)},
    {"name": "width_130", "shape": (1, 1, 1, 130)},
    {"name": "rank4_guard", "shape": (1, 5, 4, 128)},
    {"name": "many_rows", "shape": (1, 1, 1024, 64)},
    {"name": "distinct_rows", "shape": (1, 1, 5, 256), "input_fn": _distinct_rows},
    {"name": "single_row", "shape": (1, 1, 1, 896)},
    {"name": "mixed_sign", "shape": (1, 1, 4, 128), "input_fn": _mixed_sign},
    {"name": "large_4096", "shape": (1, 1, 1, 4096)},
    {"name": "large_8192", "shape": (1, 1, 1, 8192)},
    {
        "name": "weight_zeros_neg",
        "shape": (1, 1, 1, 128),
        "weight_fn": _weight_zeros_neg,
    },
]


def export_rms_norm_cases(out_dir: str) -> None:
    """Export every coverage case plus its torch golden for the native test.

    Writes `<name>.pte`, `<name>.input.bin`, `<name>.golden.bin` (raw little-endian
    fp32) under `out_dir` for each case in `_CASES`.
    """
    os.makedirs(out_dir, exist_ok=True)
    for case in _CASES:
        shape = case["shape"]
        hidden = shape[-1]
        weight_fn = case.get("weight_fn", _linspace_weight)
        input_fn = case.get("input_fn", _ramp)

        model = RmsNormModule(hidden, eps=1e-6)
        with torch.no_grad():
            model.weight.copy_(weight_fn(hidden))
        x = input_fn(shape)
        with torch.no_grad():
            golden = model(x)

        ep = torch.export.export(model, (x,))
        et_program = to_edge_transform_and_lower(
            ep, partitioner=[VulkanPartitioner()]
        ).to_executorch()

        name = case["name"]
        with open(os.path.join(out_dir, f"{name}.pte"), "wb") as f:
            f.write(et_program.buffer)
        x.detach().cpu().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"{name}.input.bin")
        )
        golden.detach().cpu().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"{name}.golden.bin")
        )
        print(f"Exported case {name} {tuple(shape)}")


if __name__ == "__main__":
    unittest.main()
