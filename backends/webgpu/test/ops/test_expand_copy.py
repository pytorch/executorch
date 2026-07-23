# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.expand_copy.default` export + golden for the WebGPU backend.

Exports single-op expand-copy graphs through VulkanPartitioner and checks an fp64
torch golden. expand_copy materializes a broadcasted view (size-1 input dims, and
rank-increasing leading dims) into the target shape via a pure gather -- no
arithmetic, so fp64 and fp32 agree exactly. Configs cover a broadcast leading
dim, a broadcast middle dim, and a rank increase (input rank < output rank) that
exercises the kernel's right-alignment of input dims into the output rank.
"""

from __future__ import annotations

import math
import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (input shape, expanded shape).
CONFIGS = {
    "broadcast_leading": ((1, 4), (3, 4)),
    "broadcast_middle": ((2, 1, 5), (2, 4, 5)),
    "rank_increase": ((4,), (3, 4)),
}


class ExpandCopyModule(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.expand(self.shape).clone()


def _det_input(shape: tuple[int, ...]) -> torch.Tensor:
    """Deterministic fp32 ramp; a broadcast dim repeats it so the copy is visible."""
    return torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)


def _export(m: torch.nn.Module, x: torch.Tensor):
    ep = torch.export.export(m, (x,))
    return to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()


def _delegates(et) -> bool:
    return any(
        d.id == "VulkanBackend"
        for plan in et.executorch_program.execution_plan
        for d in plan.delegates
    )


def _top_level_op_names(et) -> set[str]:
    return {
        op.name
        for plan in et.executorch_program.execution_plan
        for op in plan.operators
    }


class TestExpandCopy(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (in_shape, out_shape) in CONFIGS.items():
            with self.subTest(name=name):
                x = _det_input(in_shape)
                et = _export(ExpandCopyModule(out_shape).eval(), x)
                self.assertTrue(
                    _delegates(et),
                    f"Expected a VulkanBackend delegate (expand_copy {name})",
                )
                # Delegated => expand_copy absent from top-level portable ops.
                self.assertFalse(
                    any("expand_copy" in n for n in _top_level_op_names(et)),
                    f"expand_copy leaked into top-level ops ({name})",
                )

    def test_golden_matches_torch(self) -> None:
        for name, (in_shape, out_shape) in CONFIGS.items():
            with self.subTest(name=name):
                x = _det_input(in_shape)
                golden = x.double().expand(out_shape).clone()
                got = ExpandCopyModule(out_shape)(x)
                torch.testing.assert_close(got.double(), golden)


def export_expand_copy_model(
    out_shape: tuple[int, ...],
    in_shape: tuple[int, ...],
    pte_path: str,
    golden_path: str,
    input_path: str,
) -> None:
    """Write an expand_copy .pte + torch golden (raw LE fp32) + raw LE fp32 input."""
    m = ExpandCopyModule(out_shape).eval()
    x = _det_input(in_shape)
    golden = m(x).detach().numpy().astype("<f4")
    et = _export(m, x)
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    golden.tofile(golden_path)
    x.numpy().astype("<f4").tofile(input_path)
    print(
        f"Exported {pte_path}; golden {golden_path} ({golden.size} floats); "
        f"input {input_path} ({x.numel()} floats)"
    )


if __name__ == "__main__":
    unittest.main()
