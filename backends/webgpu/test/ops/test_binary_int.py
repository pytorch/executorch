# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


# The integer operands are derived from the float input rather than from
# torch.arange or full_like, neither of which this backend runs on int, so these
# stay self-contained.
class AddIntModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = (x * 8.0).to(torch.int32)
        b = (x * 2.0).to(torch.int32)
        return (a + b).to(torch.float32)


class SubIntModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = (x * 8.0).to(torch.int32)
        b = (x * 2.0).to(torch.int32)
        return (a - b).to(torch.float32)


class MulIntModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = (x * 8.0).to(torch.int32)
        b = (x * 2.0).to(torch.int32)
        return (a * b).to(torch.float32)


class AddAlphaIntModule(torch.nn.Module):
    """alpha is baked in as an i32 pipeline override on the integer path."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = (x * 8.0).to(torch.int32)
        b = (x * 2.0).to(torch.int32)
        return torch.add(a, b, alpha=3).to(torch.float32)


class SubAlphaIntModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = (x * 8.0).to(torch.int32)
        b = (x * 2.0).to(torch.int32)
        return torch.sub(a, b, alpha=3).to(torch.float32)


class BroadcastIntModule(torch.nn.Module):
    """Broadcast path of the int shader: the bicubic index math combines an
    [N, 1] row index with an [N] column index."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = (x[:, :1] * 8.0).to(torch.int32)
        cols = (x[0] * 8.0).to(torch.int32)
        return (rows * cols).to(torch.float32)


def binary_int_factory(variant: str = "add") -> torch.nn.Module:
    return {
        "add": AddIntModule,
        "sub": SubIntModule,
        "mul": MulIntModule,
        "add_alpha": AddAlphaIntModule,
        "sub_alpha": SubAlphaIntModule,
        "broadcast": BroadcastIntModule,
    }[variant]()


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
    fp32 shaders. Delegation alone cannot catch that -- the fp32 path accepted
    these graphs too -- so the numeric coverage lives in the `binary_int`
    op-test suite; these only pin that the ops stay inside the delegate."""

    def _check(self, variant: str, op_substr: str, shape=(16,)) -> None:
        example_inputs = (torch.randn(*shape),)
        ep = torch.export.export(binary_int_factory(variant), example_inputs)
        edge = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
        self.assertTrue(_delegated(edge.to_executorch()))
        self.assertTrue(_op_delegated(edge, op_substr))

    def test_add_int_delegates(self) -> None:
        self._check("add", "add.Tensor")

    def test_sub_int_delegates(self) -> None:
        self._check("sub", "sub.Tensor")

    def test_mul_int_delegates(self) -> None:
        self._check("mul", "mul.Tensor")

    def test_add_alpha_int_delegates(self) -> None:
        self._check("add_alpha", "add.Tensor")

    def test_sub_alpha_int_delegates(self) -> None:
        self._check("sub_alpha", "sub.Tensor")

    def test_broadcast_int_delegates(self) -> None:
        self._check("broadcast", "mul.Tensor", shape=(4, 4))


if __name__ == "__main__":
    unittest.main()
