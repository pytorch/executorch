# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.sum.dim_IntList` / `aten.mean.dim` single-dim reduction export + fp64 golden.

Exports single-op sum/mean graphs through VulkanPartitioner and checks the kernel
math against an fp64 torch reference. The handler reduces one dim at a time via an
outer/r/inner decomposition: `dim=-1` gives inner=1 (unit-stride reduction), a
middle dim gives inner>1 (the non-unit-stride path); `keepdim` toggles whether the
reduced dim survives in the output shape.

`AmaxModule`/`AminModule` (below) are imported by `cases.py` for the amax/amin
op-test suites. The WebGPU backend supports only the last-dim (per-row) reduction
on buffer storage, so those reduce over `dim=-1`, mirroring Vulkan's per-row path.
"""

from __future__ import annotations

import math
import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.backend.utils import get_delegates, get_non_lowered_nodes


class ReduceModule(torch.nn.Module):
    def __init__(self, op: str, dim: int, keepdim: bool) -> None:
        super().__init__()
        self.op = op
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.op == "sum":
            return torch.sum(x, dim=self.dim, keepdim=self.keepdim)
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


# (name, shape, dim, keepdim): dim=-1 -> inner=1; middle dim -> inner>1.
CONFIGS = [
    ("last_dim_keep", (4, 8), -1, True),
    ("last_dim_drop", (4, 8), -1, False),
    ("middle_dim_drop", (2, 3, 4), 1, False),  # inner=4: non-unit-stride reduction
    ("middle_dim_keep", (2, 3, 4), 1, True),
]


def _det_input(shape) -> torch.Tensor:
    """Deterministic fp32 [shape]; the C++ side reconstructs it bit-for-bit.

    v[flat] = ((flat % 17) - 8) / 16 -- exact in fp32 (small modulus, po2 denominator).
    """
    n = 1
    for s in shape:
        n *= s
    flat = torch.arange(n, dtype=torch.float32)
    return ((flat % 17) - 8).div(16.0).reshape(shape)


# Shared structural and manifest-driven extrema case authority.
EXTREMA_CONFIGS = (
    ("keepdim_2d", (37, 41), -1, True, "default"),
    ("nodim_2d", (37, 41), -1, False, "default"),
    ("keepdim_3d", (5, 7, 11), -1, True, "default"),
    ("nodim_3d", (5, 7, 11), -1, False, "default"),
    ("sign_trap_63_drop", (3, 63), -1, False, "sign_trap"),
    ("tie_64_keep", (2, 64), -1, True, "tie"),
    ("tie_65_posdim_drop", (2, 65), 1, False, "tie"),
    ("sign_trap_255_keep", (2, 255), -1, True, "sign_trap"),
    ("tie_256_drop", (2, 256), -1, False, "tie"),
    ("tie_257_posdim_keep", (2, 257), 1, True, "tie"),
)


def _sign_trap_input(shape, *, for_max: bool) -> torch.Tensor:
    magnitude = ((torch.arange(math.prod(shape), dtype=torch.float32) % 31) + 1).div(
        8.0
    )
    rows = magnitude.reshape(-1, shape[-1])
    for row_index, row in enumerate(rows):
        row.add_(float(row_index))
    values = -magnitude if for_max else magnitude
    return values.reshape(shape)


def amax_sign_trap_input(shape) -> torch.Tensor:
    return _sign_trap_input(shape, for_max=True)


def amin_sign_trap_input(shape) -> torch.Tensor:
    return _sign_trap_input(shape, for_max=False)


def _tie_input(shape, *, for_max: bool) -> torch.Tensor:
    values = ((torch.arange(math.prod(shape), dtype=torch.float32) % 29) - 14).div(8.0)
    rows = values.reshape(-1, shape[-1])
    for row_index, row in enumerate(rows):
        extreme = 16.0 + float(row_index)
        if not for_max:
            extreme = -extreme
        row[-2] = extreme
        row[-1] = extreme
    return values.reshape(shape)


def amax_tie_input(shape) -> torch.Tensor:
    return _tie_input(shape, for_max=True)


def amin_tie_input(shape) -> torch.Tensor:
    return _tie_input(shape, for_max=False)


def _extrema_input(op: str, input_class: str, shape) -> torch.Tensor:
    if input_class == "default":
        return _det_input(shape)
    if input_class == "sign_trap":
        return (
            amax_sign_trap_input(shape) if op == "amax" else amin_sign_trap_input(shape)
        )
    return amax_tie_input(shape) if op == "amax" else amin_tie_input(shape)


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


def _fp64_golden(x: torch.Tensor, op: str, dim: int, keepdim: bool) -> torch.Tensor:
    xd = x.double()
    if op == "sum":
        ref = torch.sum(xd, dim=dim, keepdim=keepdim)
    else:
        ref = torch.mean(xd, dim=dim, keepdim=keepdim)
    return ref.to(torch.float32)


class TestReduce(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for op in ("sum", "mean"):
            for name, shape, dim, keepdim in CONFIGS:
                with self.subTest(op=op, config=name):
                    x = _det_input(shape)
                    et = _export(ReduceModule(op, dim, keepdim).eval(), x)
                    self.assertTrue(
                        _delegates(et),
                        f"Expected a VulkanBackend delegate ({op} {name})",
                    )

    def test_matches_fp64_golden(self) -> None:
        for op in ("sum", "mean"):
            for name, shape, dim, keepdim in CONFIGS:
                with self.subTest(op=op, config=name):
                    x = _det_input(shape)
                    got = ReduceModule(op, dim, keepdim)(x)
                    golden = _fp64_golden(x, op, dim, keepdim)
                    torch.testing.assert_close(got, golden, atol=5e-4, rtol=1e-3)


def export_reduce_model(
    op: str,
    shape,
    dim: int,
    keepdim: bool,
    pte_path: str,
    golden_path: str,
    input_path: str,
) -> None:
    """Write a reduce .pte + torch fp64 golden (raw LE fp32) + raw LE fp32 input."""
    m = ReduceModule(op, dim, keepdim).eval()
    x = _det_input(shape)
    et = _export(m, x)
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    _fp64_golden(x, op, dim, keepdim).numpy().astype("<f4").tofile(golden_path)
    x.numpy().astype("<f4").tofile(input_path)
    print(f"Exported {pte_path}; golden {golden_path}; input {input_path}")


class AmaxModule(torch.nn.Module):
    def __init__(self, keepdim: bool, dim: int = -1) -> None:
        super().__init__()
        self.keepdim = keepdim
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=self.dim, keepdim=self.keepdim)


class AminModule(torch.nn.Module):
    def __init__(self, keepdim: bool, dim: int = -1) -> None:
        super().__init__()
        self.keepdim = keepdim
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amin(x, dim=self.dim, keepdim=self.keepdim)


class TestExtrema(unittest.TestCase):
    def test_config_contract(self) -> None:
        self.assertEqual(
            EXTREMA_CONFIGS,
            (
                ("keepdim_2d", (37, 41), -1, True, "default"),
                ("nodim_2d", (37, 41), -1, False, "default"),
                ("keepdim_3d", (5, 7, 11), -1, True, "default"),
                ("nodim_3d", (5, 7, 11), -1, False, "default"),
                ("sign_trap_63_drop", (3, 63), -1, False, "sign_trap"),
                ("tie_64_keep", (2, 64), -1, True, "tie"),
                ("tie_65_posdim_drop", (2, 65), 1, False, "tie"),
                ("sign_trap_255_keep", (2, 255), -1, True, "sign_trap"),
                ("tie_256_drop", (2, 256), -1, False, "tie"),
                ("tie_257_posdim_keep", (2, 257), 1, True, "tie"),
            ),
        )

    def test_exports_fully_delegated(self) -> None:
        for op, module_cls in (("amax", AmaxModule), ("amin", AminModule)):
            for name, shape, dim, keepdim, input_class in EXTREMA_CONFIGS:
                with self.subTest(op=op, config=name):
                    x = _extrema_input(op, input_class, shape)
                    ep = torch.export.export(module_cls(keepdim, dim).eval(), (x,))
                    edge = to_edge_transform_and_lower(
                        ep, partitioner=[VulkanPartitioner()]
                    )
                    graph = edge.exported_program().graph_module.graph
                    self.assertEqual(len(get_delegates(graph)), 1)
                    self.assertEqual(get_non_lowered_nodes(graph), [])


if __name__ == "__main__":
    unittest.main()
