# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.slice_copy.Tensor` module + configs for the WebGPU op-test framework.

`SliceModule` + `CONFIGS` are imported by `cases.py` to drive the declarative
op-test suite. `TestSlice` is the export-delegation + eager-correctness smoke
test.
"""

import unittest

import torch

from executorch.backends.vulkan import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (input_shape, slice_fn)
CONFIGS = {
    "dim1_1_5": ((3, 8, 4), lambda x: x[:, 1:5]),
    "lastdim": ((3, 8, 4), lambda x: x[..., 1:3]),
    "step2": ((3, 8, 4), lambda x: x[:, 0:8:2]),
    "neg_end": ((3, 8, 4), lambda x: x[:, 1:-1]),
}


class SliceModule(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


def _det_input(shape):
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _export(fn, x: torch.Tensor):
    ep = torch.export.export(SliceModule(fn).eval(), (x,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class TestSlice(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (shape, fn) in CONFIGS.items():
            et = _export(fn, _det_input(shape))
            self.assertTrue(
                _delegated(et), f"Expected a VulkanBackend delegate (slice {name})"
            )

    def test_golden_matches_eager(self) -> None:
        for _, (shape, fn) in CONFIGS.items():
            x = _det_input(shape)
            torch.testing.assert_close(SliceModule(fn)(x), fn(x))


if __name__ == "__main__":
    unittest.main()
