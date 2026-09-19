# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class ClampIntModule(torch.nn.Module):
    """Integer clamp with both bounds, as index arithmetic produces."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(0, 16, 1, dtype=torch.int32)
        return x + torch.clamp(idx, 2, 11).to(torch.float32)


class ClampIntMinOnlyModule(torch.nn.Module):
    """Only a lower bound; the upper becomes None and must saturate to INT32_MAX."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(0, 16, 1, dtype=torch.int32)
        return x + torch.clamp(idx, min=5).to(torch.float32)


class ClampIntMaxOnlyModule(torch.nn.Module):
    """Only an upper bound; the lower becomes None and must saturate to INT32_MIN."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(0, 16, 1, dtype=torch.int32)
        return x + torch.clamp(idx, max=9).to(torch.float32)


class ClampIntLargeBoundModule(torch.nn.Module):
    """Bound above 2^24, which float cannot represent exactly. Pins that the i32
    bounds reach the shader without a detour through float."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(0, 16, 1, dtype=torch.int32)
        return x + torch.clamp(idx, 0, 16777217).to(torch.float32)


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


class TestClampInt(unittest.TestCase):
    """Integer aten.clamp.default export tests. fp32 clamp is covered by the
    declarative unary suite; these pin the int32 shader variant, which the ViT
    positional-encoding path needs."""

    def _check(self, model, example_inputs) -> None:
        ep = torch.export.export(model, example_inputs)
        edge = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
        et = edge.to_executorch()
        self.assertTrue(_delegated(et), "Expected a VulkanBackend delegate")
        self.assertTrue(
            _op_delegated(edge, "clamp"),
            "Expected aten.clamp to be absorbed into the delegate, not left on CPU",
        )

    def test_clamp_int_both_bounds(self) -> None:
        self._check(ClampIntModule(), (torch.randn(16),))

    def test_clamp_int_min_only(self) -> None:
        self._check(ClampIntMinOnlyModule(), (torch.randn(16),))

    def test_clamp_int_max_only(self) -> None:
        self._check(ClampIntMaxOnlyModule(), (torch.randn(16),))

    def test_clamp_int_bound_above_float_precision(self) -> None:
        self._check(ClampIntLargeBoundModule(), (torch.randn(16),))


if __name__ == "__main__":
    unittest.main()
