# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.convolution.default` (conv2d + conv_transpose2d) module + inputs.

`make_conv` and `_chw_ramp` are imported by `cases.py` to drive the declarative
op-test suite. `Conv2dTest` is the export-delegation smoke test. conv2d is the
DaViT patch-embed / downsample op in Florence-2; conv_transpose2d shares the
same `aten.convolution.default` registration (folded by the `transposed` arg).
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class Conv2dModule(torch.nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        transposed: bool = False,
    ):
        super().__init__()
        cls = torch.nn.ConvTranspose2d if transposed else torch.nn.Conv2d
        self.conv = cls(
            in_ch, out_ch, kernel, stride=stride, padding=padding, groups=groups
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def make_conv(
    in_ch: int,
    out_ch: int,
    kernel: int,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    transposed: bool = False,
) -> torch.nn.Module:
    """Factory with deterministic small weights/bias for reproducible goldens."""
    m = Conv2dModule(in_ch, out_ch, kernel, stride, padding, groups, transposed)
    with torch.no_grad():
        w = m.conv.weight
        w.copy_(
            torch.linspace(-1.0, 1.0, w.numel(), dtype=torch.float32).reshape(w.shape)
            / (kernel * kernel)
        )
        if m.conv.bias is not None:
            b = m.conv.bias
            b.copy_(torch.linspace(-0.5, 0.5, b.numel(), dtype=torch.float32))
    return m


def _chw_ramp(shape) -> torch.Tensor:
    """Deterministic [N, C, H, W] ramp in [-1, 1]."""
    n = 1
    for d in shape:
        n *= d
    return torch.linspace(-1.0, 1.0, n, dtype=torch.float32).reshape(shape)


def _export(m: torch.nn.Module, x: torch.Tensor):
    ep = torch.export.export(m, (x,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


class Conv2dTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        configs = [
            ("conv", make_conv(3, 8, 3, padding=1), (1, 3, 8, 8)),
            ("transpose", make_conv(4, 4, 2, stride=2, transposed=True), (1, 4, 4, 4)),
        ]
        for name, model, shape in configs:
            et = _export(model.eval(), _chw_ramp(shape))
            found = any(
                d.id == "VulkanBackend"
                for plan in et.executorch_program.execution_plan
                for d in plan.delegates
            )
            self.assertTrue(found, f"Expected a VulkanBackend delegate (conv {name})")
