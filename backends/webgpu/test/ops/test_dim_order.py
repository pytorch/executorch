# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`dim_order_ops._clone_dim_order.default` export delegation for the WebGPU backend.

`torch.clone(x)` lowers to `dim_order_ops._clone_dim_order.default` (the edge
dialect's dim-order-aware clone). On the buffer-only WebGPU backend it is a
numel-preserving flat copy handled by the shared `add_flat_copy` DMA helper (no
WGSL). The partitioner tags it (single-node partitions are allowed), so a
`VulkanBackend` delegate is formed; `RemoveRedundantOpsTransform` then folds the
identity clone out of the delegate in preprocess, so it never reaches the native
runtime (hence no `.golden.bin` / native sweep — like `clone`).

`test_export_delegates` locks the contract that the op is absorbed into the
delegate and never survives as a top-level portable (CPU-fallback) node;
`test_golden_matches_eager` locks the fp64 `x.clone()` reference. Configs cover
1D, 3D, and 4D contiguous inputs.
"""

from __future__ import annotations

import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
    VulkanPartitioner,
)
from executorch.exir import to_edge_transform_and_lower

# name -> input_shape
CONFIGS = {
    "flat": (16,),
    "3d": (2, 3, 4),
    "4d": (1, 3, 4, 5),
}


class CloneDimOrderModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clone(x)


def _det_input(shape):
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _lower(x: torch.Tensor):
    ep = torch.export.export(CloneDimOrderModule().eval(), (x,))
    return to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])


def _delegates(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


def _op_absent_from_toplevel(edge, op_substr: str) -> bool:
    # Delegated/folded ops are absorbed; none may survive as a top-level node.
    gm = edge.exported_program().graph_module
    return all(op_substr not in str(getattr(n, "target", "")) for n in gm.graph.nodes)


class TestCloneDimOrder(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, shape in CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower(_det_input(shape))
                et = edge.to_executorch()
                self.assertTrue(
                    _delegates(et),
                    f"Expected a VulkanBackend delegate (clone_dim_order {name})",
                )
                self.assertTrue(
                    _op_absent_from_toplevel(edge, "_clone_dim_order"),
                    f"_clone_dim_order left as a top-level portable op for {name}",
                )

    def test_golden_matches_eager(self) -> None:
        for name, shape in CONFIGS.items():
            with self.subTest(name=name):
                x = _det_input(shape)
                ref = x.to(torch.float64).clone()
                torch.testing.assert_close(
                    CloneDimOrderModule()(x).to(torch.float64), ref
                )


if __name__ == "__main__":
    unittest.main()
