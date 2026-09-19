# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class AddIntModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(0, 16, 1, dtype=torch.int32)
        return x + (idx + torch.full_like(idx, 3)).to(torch.float32)


class SubIntModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(0, 16, 1, dtype=torch.int32)
        return x + (idx - torch.full_like(idx, 2)).to(torch.float32)


class MulIntModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(0, 16, 1, dtype=torch.int32)
        return x + (idx * torch.full_like(idx, 3)).to(torch.float32)


class BroadcastIntModule(torch.nn.Module):
    """Broadcast path of the int shader: the bicubic index math combines a
    [N, 1] row index with a [N] column index."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(0, 4, 1, dtype=torch.int32).reshape(4, 1)
        cols = torch.arange(0, 4, 1, dtype=torch.int32)
        return x + (rows * 4 + cols).to(torch.float32)


class IndexMathModule(torch.nn.Module):
    """The shape the ViT positional-encoding interpolation actually produces:
    integer index arithmetic feeding a clamp. Routing any of these through the
    fp32 shader reads the integer bit pattern as a denormal, so the result
    collapses to ~0 rather than failing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(0, 16, 1, dtype=torch.int32)
        lo = torch.clamp(idx - torch.full_like(idx, 1), 0, 36)
        hi = torch.clamp(idx + torch.full_like(idx, 2), 0, 36)
        return x + (lo + hi).to(torch.float32)


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


def _op_delegated(edge, op_substr: str) -> bool:
    from executorch.exir.lowered_backend_module import get_lowered_submodules

    gm = edge.exported_program().graph_module
    if any(op_substr in str(getattr(n, "target", "")) for n in gm.graph.nodes):
        return False
    return any(
        op_substr in str(getattr(dn, "target", ""))
        for _, lowered, _ in get_lowered_submodules(gm)
        for dn in lowered.original_module.graph_module.graph.nodes
    )


class TestBinaryInt(unittest.TestCase):
    """Integer add/sub/mul export tests. Both f32 and i32 are 4 bytes, so the
    byte-size guard these handlers used to rely on let integer tensors reach the
    fp32 shaders; these pin the i32 shader variants."""

    def _check(self, model, op_substr: str) -> None:
        example_inputs = (torch.randn(16),)
        ep = torch.export.export(model, example_inputs)
        edge = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
        et = edge.to_executorch()
        self.assertTrue(_delegated(et))
        self.assertTrue(_op_delegated(edge, op_substr))

    def test_add_int_delegates(self) -> None:
        self._check(AddIntModule(), "add.Tensor")

    def test_sub_int_delegates(self) -> None:
        self._check(SubIntModule(), "sub.Tensor")

    def test_mul_int_delegates(self) -> None:
        self._check(MulIntModule(), "mul.Tensor")

    def test_broadcast_int_delegates(self) -> None:
        example_inputs = (torch.randn(4, 4),)
        ep = torch.export.export(BroadcastIntModule(), example_inputs)
        edge = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
        self.assertTrue(_delegated(edge.to_executorch()))
        self.assertTrue(_op_delegated(edge, "mul.Tensor"))

    def test_index_math_delegates(self) -> None:
        self._check(IndexMathModule(), "clamp")


if __name__ == "__main__":
    unittest.main()
