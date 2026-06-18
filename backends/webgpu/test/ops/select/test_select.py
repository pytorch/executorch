# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.select_copy.int` module + configs for the WebGPU op-test framework.

`SelectModule` + `CONFIGS` are imported by `cases.py` to drive the declarative
op-test suite. `TestSelect` is the export-delegation + eager-correctness smoke test.
Configs cover the leading, middle, and last dim plus a negative index (output rank =
input rank - 1).
"""

import unittest

import torch

from executorch.backends.vulkan import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (input_shape, select_fn)
CONFIGS = {
    "dim0": ((3, 8, 4), lambda x: x[1]),
    "middle": ((3, 8, 4), lambda x: x[:, 2]),
    "last": ((3, 8, 4), lambda x: x[..., 3]),
    "neg_idx": ((3, 8, 4), lambda x: x[:, -1]),
}


class SelectModule(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


def _det_input(shape):
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _export(fn, x: torch.Tensor):
    ep = torch.export.export(SelectModule(fn).eval(), (x,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class TestSelect(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (shape, fn) in CONFIGS.items():
            et = _export(fn, _det_input(shape))
            self.assertTrue(
                _delegated(et), f"Expected a VulkanBackend delegate (select {name})"
            )

    def test_golden_matches_eager(self) -> None:
        for _, (shape, fn) in CONFIGS.items():
            x = _det_input(shape)
            torch.testing.assert_close(SelectModule(fn)(x), fn(x))


if __name__ == "__main__":
    unittest.main()
