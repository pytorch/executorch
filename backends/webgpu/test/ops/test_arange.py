# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class ArangeModule(torch.nn.Module):
    """x + arange(start, end, step, dtype).

    The arange is baked into the graph as a constant-producing node; adding it to
    the input keeps the result dependent on it so the value is actually checked.
    `dtype=None` follows torch's own defaulting: float literals give fp32, integer
    literals give int64 (which the pipeline downcasts to int32 on device).
    """

    def __init__(self, start, end, step, dtype=None) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.step = step
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.arange(self.start, self.end, self.step, dtype=self.dtype)
        return x + a.to(torch.float32)


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


def _op_delegated(edge, op_substr: str) -> bool:
    # The op must be absorbed into a delegate: absent from the top-level graph AND
    # present inside a lowered submodule reached by an executorch_call_delegate node
    # (a bare absence check also passes for an empty graph or a renamed op).
    from executorch.exir.lowered_backend_module import get_lowered_submodules

    gm = edge.exported_program().graph_module
    if any(op_substr in str(getattr(n, "target", "")) for n in gm.graph.nodes):
        return False
    return any(
        op_substr in str(getattr(dn, "target", ""))
        for _, lowered, _ in get_lowered_submodules(gm)
        for dn in lowered.original_module.graph_module.graph.nodes
    )


class TestArange(unittest.TestCase):
    """aten.arange.start_step export tests — uses VulkanPartitioner since the
    WebGPU runtime directly consumes the Vulkan delegate (VK00 FlatBuffer).
    Numeric coverage lives in the op_tests suite, which executes on device."""

    def _check(self, model, example_inputs) -> None:
        ep = torch.export.export(model, example_inputs)
        edge = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
        et = edge.to_executorch()
        self.assertTrue(_delegated(et), "Expected a VulkanBackend delegate")
        self.assertTrue(
            _op_delegated(edge, "arange"),
            "Expected aten.arange to be absorbed into the delegate, not left on CPU",
        )

    def test_arange_float(self) -> None:
        self._check(ArangeModule(0.0, 16.0, 1.0), (torch.randn(16),))

    def test_arange_int32(self) -> None:
        # Explicit int32 — exercises the i32 shader variant directly.
        self._check(ArangeModule(0, 16, 1, torch.int32), (torch.randn(16),))

    def test_arange_int_default(self) -> None:
        # Integer literals without dtype trace as int64; the pipeline downcasts
        # to int32 on device. This is the shape DINOv2's positional-encoding
        # path actually produces, so cover it alongside the explicit case.
        self._check(ArangeModule(0, 16, 1), (torch.randn(16),))

    def test_arange_step(self) -> None:
        self._check(ArangeModule(3.0, 35.0, 2.0), (torch.randn(16),))


if __name__ == "__main__":
    unittest.main()
