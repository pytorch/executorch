# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`et_vk.apply_rotary_emb_interleaved` module + configs for the op-test framework.

`RopeInterleavedModule` calls the custom op directly (EdgeTAM has no aten
lowering / fusion pattern for it); both x and the [cos,sin] freqs are fp32
runtime inputs and the op has a CPU eager impl, so the framework goldens it
directly. `RopeInterleavedTest` is the export-delegation smoke test.
"""

import unittest

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> input_shape (B, N, C), freqs_shape (N, C)
CONFIGS = {
    "bnc": ((1, 4, 8), (4, 8)),
    "batch": ((2, 3, 8), (3, 8)),
    "c4": ((1, 5, 4), (5, 4)),
    "c16": ((1, 2, 16), (2, 16)),
}


class RopeInterleavedModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return torch.ops.et_vk.apply_rotary_emb_interleaved(x, freqs)


def _det(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(in_shape, freqs_shape):
    x = _det(in_shape, 1)
    freqs = _det(freqs_shape, 2)
    ep = torch.export.export(RopeInterleavedModule().eval(), (x, freqs))
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


class RopeInterleavedTest(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (in_shape, freqs_shape) in CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower(in_shape, freqs_shape)
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (rope_interleaved {name})",
                )
                self.assertTrue(
                    _op_delegated(edge, "interleaved"),
                    f"rope_interleaved not delegated (CPU fallback) for {name}",
                )
