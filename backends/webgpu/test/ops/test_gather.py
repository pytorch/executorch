# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.gather.default` export + fp64 golden for the WebGPU backend.

Exports single-op gather graphs through VulkanPartitioner and writes a torch-computed
golden (the native binary has no ATen). gather(self, dim, index) copies self along
`dim` at the positions named by index (out has index's shape). Configs cover the last
dim, dim 0, and a rank-3 negative dim (exercises the handler's dim-normalization). The
native test reconstructs the deterministic inputs bit-for-bit.
"""

from __future__ import annotations

import os
import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (self_shape, dim, index_shape). Ranks stay <= 4 (TensorMeta MAX_NDIM).
CONFIGS = {
    "cols": ((4, 8), 1, (4, 3)),
    "rows": ((5, 6), 0, (3, 6)),
    "rank3_neg": ((2, 3, 4), -1, (2, 3, 2)),
}


class GatherModule(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return torch.gather(x, self.dim, index)


def _det_inputs(self_shape, dim: int, index_shape):
    """Distinct fp32 source (a wrong pick is visible) + in-range int64 index."""
    n = 1
    for s in self_shape:
        n *= s
    x = torch.arange(n, dtype=torch.float32).reshape(self_shape)
    bound = self_shape[dim]
    m = 1
    for s in index_shape:
        m *= s
    index = (torch.arange(m, dtype=torch.int64) % bound).reshape(index_shape)
    return x, index


def _lower(m: torch.nn.Module, x: torch.Tensor, index: torch.Tensor):
    ep = torch.export.export(m, (x, index))
    return to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])


def _delegated(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


def _op_delegated(edge) -> bool:
    # gather must be absorbed into the delegate, not a top-level CPU node.
    gm = edge.exported_program().graph_module
    return all("gather" not in str(getattr(n, "target", "")) for n in gm.graph.nodes)


class TestGather(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (self_shape, dim, index_shape) in CONFIGS.items():
            with self.subTest(name=name):
                x, index = _det_inputs(self_shape, dim, index_shape)
                edge = _lower(GatherModule(dim).eval(), x, index)
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (gather {name})",
                )
                self.assertTrue(
                    _op_delegated(edge),
                    f"gather not delegated (fell back to CPU) for {name}",
                )

    def test_op_matches_fp64_golden(self) -> None:
        for name, (self_shape, dim, index_shape) in CONFIGS.items():
            with self.subTest(name=name):
                x, index = _det_inputs(self_shape, dim, index_shape)
                got = GatherModule(dim)(x, index)
                golden = torch.gather(x.double(), dim, index).to(torch.float32)
                torch.testing.assert_close(got, golden)


def export_gather_model(name: str, pte_path: str, golden_path: str) -> None:
    """Write one config's gather .pte + fp64 torch golden (raw LE fp32)."""
    self_shape, dim, index_shape = CONFIGS[name]
    x, index = _det_inputs(self_shape, dim, index_shape)
    et = _lower(GatherModule(dim).eval(), x, index).to_executorch()
    golden = torch.gather(x.double(), dim, index).to(torch.float32)
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    golden.numpy().astype("<f4").tofile(golden_path)
    print(f"Exported {pte_path}; golden {golden_path} ({golden.numel()} floats)")


def export_all_gather_models(out_dir: str) -> None:
    for name in CONFIGS:
        export_gather_model(
            name,
            os.path.join(out_dir, f"gather_{name}.pte"),
            os.path.join(out_dir, f"gather_{name}.golden.bin"),
        )


if __name__ == "__main__":
    unittest.main()
