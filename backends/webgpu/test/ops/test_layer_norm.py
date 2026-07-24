# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.native_layer_norm.default` module + inputs for the WebGPU op-test framework.

`LayerNormModule`, `make_layer_norm`, and `_ramp` are imported by `cases.py` to
drive the declarative op-test suite. `LayerNormTest` is the export-delegation
smoke test. LayerNorm is pervasive in BART + the DaViT vision encoder
(Florence-2); both the affine (weight+bias) and the no-affine path are covered.
"""

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class LayerNormModule(torch.nn.Module):
    """LayerNorm over the last dim; lowers to aten.native_layer_norm.default."""

    def __init__(self, normalized_shape: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.ln = torch.nn.LayerNorm(
            normalized_shape, eps=eps, elementwise_affine=affine
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


def make_layer_norm(
    normalized_shape: int, affine: bool = True, eps: float = 1e-5
) -> torch.nn.Module:
    """Factory with deterministic non-trivial affine params (when affine)."""
    m = LayerNormModule(normalized_shape, affine=affine, eps=eps)
    if affine:
        with torch.no_grad():
            m.ln.weight.copy_(
                torch.linspace(0.5, 1.5, normalized_shape, dtype=torch.float32)
            )
            m.ln.bias.copy_(
                torch.linspace(-0.25, 0.25, normalized_shape, dtype=torch.float32)
            )
    return m


def _ramp(shape) -> torch.Tensor:
    """Deterministic linear ramp in [-1, 1] reshaped to `shape`."""
    n = 1
    for d in shape:
        n *= d
    return torch.linspace(-1.0, 1.0, n, dtype=torch.float32).reshape(shape)


def _export(m: torch.nn.Module, x: torch.Tensor):
    ep = torch.export.export(m, (x,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


class LayerNormTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for affine in (True, False):
            et = _export(make_layer_norm(128, affine=affine).eval(), _ramp((1, 4, 128)))
            found = any(
                d.id == "VulkanBackend"
                for plan in et.executorch_program.execution_plan
                for d in plan.delegates
            )
            self.assertTrue(
                found, f"Expected a VulkanBackend delegate (layer_norm affine={affine})"
            )
