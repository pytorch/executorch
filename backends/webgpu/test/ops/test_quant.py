# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`quantize_per_tensor` + `dequantize_per_tensor` modules + configs.

Each op is tested ALONE, not as a round-trip: ET folds `dequantize(quantize(x))`
back to `x` before delegation (verified 2026-07-08), so a round-trip would test a
passthrough, not the kernels. `QuantizeModule` emits an int8 output compared
byte-exact to the torch int8 (needs the int8-golden harness path).
`DequantizeConstModule` dequantizes a BAKED int8 constant so the dequantize stage
is verified independently against torch.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

_QMIN, _QMAX = -128, 127


class QuantizeModule(torch.nn.Module):
    def __init__(self, scale: float = 0.05, zero_point: int = 0):
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized_decomposed.quantize_per_tensor(
            x, self.scale, self.zero_point, _QMIN, _QMAX, torch.int8
        )


class DequantizeConstModule(torch.nn.Module):
    def __init__(self, scale: float = 0.05, zero_point: int = 0, q_values=None):
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point
        vals = q_values if q_values is not None else list(range(-8, 8))
        self.register_buffer("q", torch.tensor(vals, dtype=torch.int8))

    def forward(self) -> torch.Tensor:
        return torch.ops.quantized_decomposed.dequantize_per_tensor(
            self.q, self.scale, self.zero_point, _QMIN, _QMAX, torch.int8
        )


def _delegates(module, inputs) -> bool:
    ep = torch.export.export(module, inputs)
    et = to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


class QuantTest(unittest.TestCase):
    def test_quantize_delegates(self) -> None:
        self.assertTrue(_delegates(QuantizeModule(), (torch.randn(4, 8),)))

    def test_dequant_const_delegates(self) -> None:
        self.assertTrue(_delegates(DequantizeConstModule(), ()))
