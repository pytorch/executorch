# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.avg_pool2d.default` module + configs for the WebGPU op-test framework.

`AvgPool2dModule` averages over a KxK window (`F.avg_pool2d`). `AvgPool2dTest`
is the export-delegation smoke test.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> input_shape, kernel, stride, padding, count_include_pad, ceil_mode, divisor
CONFIGS = {
    "basic": ((1, 2, 4, 4), [2, 2], [2, 2], [0, 0], True, False, None),
    "pad_cip": ((1, 2, 5, 5), [3, 3], [2, 2], [1, 1], True, False, None),
    "pad_nocip": ((1, 2, 5, 5), [3, 3], [2, 2], [1, 1], False, False, None),
    "asym": ((2, 3, 5, 7), [3, 2], [2, 3], [1, 1], True, False, None),
    "divisor": ((1, 1, 4, 4), [2, 2], [2, 2], [0, 0], True, False, 3),
    # ceil_mode: the last window overhangs the input -> exercises the overhang
    # divisor branch (beh/bew > 0) + the ceil output-size (3x3 vs floor 2x2).
    "ceil_cip": ((1, 1, 5, 5), [2, 2], [2, 2], [0, 0], True, True, None),
    "ceil_nocip": ((1, 2, 5, 5), [3, 3], [2, 2], [0, 0], False, True, None),
}


class AvgPool2dModule(torch.nn.Module):
    def __init__(
        self, kernel, stride, padding, count_include_pad, ceil_mode, divisor
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode
        self.divisor = divisor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(
            x,
            self.kernel,
            self.stride,
            self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor,
        )


def _det_input(shape):
    g = torch.Generator().manual_seed(1)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(cfg, x: torch.Tensor):
    _, kernel, stride, padding, cip, ceil_mode, divisor = cfg
    ep = torch.export.export(
        AvgPool2dModule(kernel, stride, padding, cip, ceil_mode, divisor).eval(), (x,)
    )
    return to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


def _op_delegated(edge, op_substr: str) -> bool:
    # op must be absorbed into the delegate, not left as a top-level CPU-fallback node.
    gm = edge.exported_program().graph_module
    return all(op_substr not in str(getattr(n, "target", "")) for n in gm.graph.nodes)


class AvgPool2dTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, cfg in CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower(cfg, _det_input(cfg[0]))
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (avg_pool2d {name})",
                )
                self.assertTrue(
                    _op_delegated(edge, "avg_pool2d.default"),
                    f"avg_pool2d not delegated (fell back to CPU) for {name}",
                )
