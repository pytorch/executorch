# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-dispatch ordering coverage for WebGPUGraph::execute().

Each model is a dependency chain whose dispatches must execute in order (one
compute pass per dispatch is the implicit barrier). Vehicle A is a single-input
add self-chain; Vehicle B chains add on a reused RmsNormModule (a heterogeneous
cross-pipeline RAW edge). Numerics are checked in test/native/test_dispatch_order.cpp.
"""

import os
import unittest

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.webgpu.test.ops.test_rms_norm import RmsNormModule
from executorch.backends.webgpu.test.tester import WEBGPU_SUPPORTED_OPS
from executorch.exir import to_edge_transform_and_lower


class ChainAddModule(torch.nn.Module):
    """z = x + x; z = z + x; ... (depth adds) -> (depth + 1) * x."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x + x
        for _ in range(self.depth - 1):
            z = z + x
        return z


class RmsNormAddModule(torch.nn.Module):
    """t = rms_norm(x); z = t + x; ... (adds adds) -- heterogeneous RAW chain."""

    def __init__(self, width: int, adds: int) -> None:
        super().__init__()
        self.rms = RmsNormModule(width, eps=1e-6)
        self.adds = adds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.rms(x) + x
        for _ in range(self.adds - 1):
            z = z + x
        return z


# (name, kind, shape, depth) -- MUST match kCases in test_dispatch_order.cpp.
_CASES = [
    ("single", "chain", (16, 16), 1),
    ("chain3", "chain", (64, 64), 3),
    ("chain5_tiny", "chain", (1, 1), 5),
    ("chain5_wide", "chain", (7, 896), 5),
    ("chain8", "chain", (256, 256), 8),
    ("deep32", "chain", (128, 128), 32),
    ("large_chain", "chain", (1024, 1024), 6),
    ("het_small", "rms", (1, 1, 7, 896), 2),
    ("het_deep", "rms", (1, 1, 5, 256), 3),
]


def _model(kind: str, shape, depth: int) -> torch.nn.Module:
    if kind == "chain":
        return ChainAddModule(depth)
    return RmsNormAddModule(shape[-1], depth)


def _lower(model: torch.nn.Module, x: torch.Tensor):
    ep = torch.export.export(model, (x,))
    return to_edge_transform_and_lower(
        ep,
        partitioner=[VulkanPartitioner(operator_allowlist=WEBGPU_SUPPORTED_OPS)],
    ).to_executorch()


class TestDispatchOrder(unittest.TestCase):
    def _assert_delegated(self, prog) -> None:
        found = any(
            d.id == "VulkanBackend"
            for p in prog.executorch_program.execution_plan
            for d in p.delegates
        )
        self.assertTrue(found, "Expected VulkanBackend delegate in .pte")

    def test_chain_add(self) -> None:
        self._assert_delegated(_lower(ChainAddModule(5), torch.randn(64, 64)))

    def test_rms_norm_add(self) -> None:
        self._assert_delegated(
            _lower(RmsNormAddModule(896, 2), torch.randn(1, 1, 7, 896))
        )


def export_dispatch_order_cases(out_dir: str) -> None:
    """Write <name>.pte, <name>.input.bin, <name>.golden.bin (raw le fp32) per case."""
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)
    for name, kind, shape, depth in _CASES:
        x = torch.randn(*shape)
        model = _model(kind, shape, depth)
        prog = _lower(model, x)
        with torch.no_grad():
            golden = model(x)
        base = os.path.join(out_dir, name)
        x.detach().cpu().numpy().astype("<f4").tofile(base + ".input.bin")
        golden.detach().cpu().numpy().astype("<f4").tofile(base + ".golden.bin")
        with open(base + ".pte", "wb") as f:
            f.write(prog.buffer)
        print(f"Exported case {name} {tuple(shape)}")


if __name__ == "__main__":
    unittest.main()
