# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten._to_copy.default` modules for the WebGPU op-test framework.

`ToCopyIntToFloatModule` / `ToCopyFloatModule` drive the export-delegation smoke
test (mirroring `test_view_copy.py`). The int32 -> fp32 numeric convert — input
int `[1, 2, 3]` -> float `[1.0, 2.0, 3.0]`, NOT the byte-reinterpretation
`0x1 -> 1.4e-45` — and the fp32 same-dtype passthrough are value-checked by the
lvp golden (yolo11n / Depth-Anything-V2).
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class ToCopyIntToFloatModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # int32 -> fp32 dtype cast (the numeric-convert path).
        return x.to(torch.float32)


class ToCopyFloatModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same-dtype copy (flat byte-copy path); copy=True keeps the op from
        # being elided as a no-op.
        return x.to(torch.float32, copy=True)


def _export(model: torch.nn.Module, x: torch.Tensor):
    ep = torch.export.export(model.eval(), (x,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class ToCopyTest(unittest.TestCase):
    def test_int_to_float_delegates(self) -> None:
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        et = _export(ToCopyIntToFloatModule(), x)
        self.assertTrue(
            _delegated(et), "Expected a VulkanBackend delegate (to_copy int->float)"
        )

    def test_float_passthrough_delegates(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        et = _export(ToCopyFloatModule(), x)
        self.assertTrue(
            _delegated(et), "Expected a VulkanBackend delegate (to_copy float->float)"
        )
