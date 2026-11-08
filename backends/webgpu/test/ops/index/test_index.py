# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`aten.index.Tensor` export + goldens for the WebGPU backend.

Exports the 1D-self advanced-index form `self[idx]` through VulkanPartitioner --
the only delegated index.Tensor (the 2D mask/freqs gathers are CPU fallbacks; see
op_registry.py:1427). It is a flat gather out[i]=self[index[i]]; the int64 index
serializes as int32 (downcast_64_bit). Distinct self values + reorder/repeat
indices make a wrong-gather bug visible. Each config writes `index_<name>.pte`,
`index_<name>.self.bin` (fp32 self), `index_<name>.idx.bin` (int32 index), and
`index_<name>.golden.bin` so the native `test_index` self-discovers them.

`gather_column_major` is a `gather` whose index input is stored column by
column, like the result of `torch.nonzero`; its `.idx.bin` holds the index in
that storage order. `scale_channels_last` scales the channels of a channels-last
input, so its result is channels-last too; its `.x.bin` and `.golden.bin` hold
that storage order.
"""

import os
import unittest

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower

# name -> (self_len, index_values)
CONFIGS = {
    "n16_m5": (16, [0, 15, 7, 7, 2]),
    "n8_rev": (8, [7, 6, 5, 4, 3, 2, 1, 0]),
    "n32_m3": (32, [31, 0, 16]),
    "n4_rep": (4, [2, 2, 2, 2, 0, 1]),
}


class IndexModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return x[idx]


class GatherModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return torch.gather(x, 0, index)


class ScaleChannelsModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * scale


def _column_major_gather_inputs():
    x = torch.arange(15, dtype=torch.float32).reshape(5, 3) * 3.0 + 0.5
    index = torch.tensor([[0, 1, 1, 2, 4], [1, 2, 0, 1, 2]]).t()
    return x, index


def _channels_last_scale_inputs():
    x = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5)
    x = x.to(memory_format=torch.channels_last)
    scale = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3, 1, 1)
    return x, scale


def _inputs(self_len, index_values):
    # Distinct self values so a wrong-index gather is visible.
    x = torch.arange(self_len, dtype=torch.float32) * 3.0 + 0.5
    idx = torch.tensor(index_values, dtype=torch.int64)
    return x, idx


def _lower(x, idx):
    ep = torch.export.export(IndexModule().eval(), (x, idx))
    return to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])


def _export(x, idx):
    return _lower(x, idx).to_executorch()


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


class TestIndex(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for name, (n, iv) in CONFIGS.items():
            with self.subTest(name=name):
                edge = _lower(*_inputs(n, iv))
                et = edge.to_executorch()
                self.assertTrue(
                    _delegated(et),
                    f"Expected a VulkanBackend delegate (index {name})",
                )
                self.assertTrue(
                    _op_delegated(edge, "index.Tensor"),
                    f"index.Tensor not delegated (fell back to CPU) for {name}",
                )

    def test_dim_order_models_delegate(self) -> None:
        for name, module, inputs, op in (
            ("gather", GatherModule(), _column_major_gather_inputs(), "gather"),
            ("scale", ScaleChannelsModule(), _channels_last_scale_inputs(), "mul"),
        ):
            with self.subTest(name=name):
                edge = to_edge_transform_and_lower(
                    torch.export.export(module.eval(), inputs),
                    partitioner=[VulkanPartitioner()],
                )
                self.assertTrue(
                    _op_delegated(edge, op),
                    f"{op} not delegated (fell back to CPU) for {name}",
                )

    def test_golden_matches_eager(self) -> None:
        for name, (n, iv) in CONFIGS.items():
            with self.subTest(name=name):
                x, idx = _inputs(n, iv)
                torch.testing.assert_close(IndexModule()(x, idx), x[idx])


def export_all_index_models(out_dir: str) -> None:
    """Write the .pte and .bin files of every model that test_index reads."""
    os.makedirs(out_dir, exist_ok=True)
    for name, (n, iv) in CONFIGS.items():
        x, idx = _inputs(n, iv)
        golden = x[idx].contiguous().detach().numpy().astype("<f4")
        et = _export(x, idx)
        base = os.path.join(out_dir, f"index_{name}")
        with open(base + ".pte", "wb") as f:
            f.write(et.buffer)
        x.numpy().astype("<f4").tofile(base + ".self.bin")
        idx.numpy().astype("<i4").tofile(base + ".idx.bin")
        golden.tofile(base + ".golden.bin")
        print(f"Exported {base}.pte; self {n} -> golden {golden.size} floats")

    x, index = _column_major_gather_inputs()
    ep = torch.export.export(GatherModule().eval(), (x, index))
    et = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
    base = os.path.join(out_dir, "gather_column_major")
    with open(base + ".pte", "wb") as f:
        f.write(et.to_executorch().buffer)
    x.numpy().astype("<f4").tofile(base + ".self.bin")
    index.t().contiguous().numpy().astype("<i4").tofile(base + ".idx.bin")
    golden = torch.gather(x, 0, index).numpy().astype("<f4")
    golden.tofile(base + ".golden.bin")
    print(f"Exported {base}.pte; golden {golden.size} floats")

    x, scale = _channels_last_scale_inputs()
    ep = torch.export.export(ScaleChannelsModule().eval(), (x, scale))
    et = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
    base = os.path.join(out_dir, "scale_channels_last")
    with open(base + ".pte", "wb") as f:
        f.write(et.to_executorch().buffer)
    x.permute(0, 2, 3, 1).numpy().astype("<f4").tofile(base + ".x.bin")
    golden = (x * scale).permute(0, 2, 3, 1).numpy().astype("<f4")
    golden.tofile(base + ".golden.bin")
    print(f"Exported {base}.pte; golden {golden.size} floats")


if __name__ == "__main__":
    unittest.main()
