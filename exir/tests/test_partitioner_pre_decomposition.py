# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from torch.export import export, ExportedProgram


class _SDPA(torch.nn.Module):
    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)


class _RecordingPartitioner(Partitioner):
    def __init__(self, name: str, calls: list[str]) -> None:
        super().__init__()
        self.name = name
        self.calls = calls
        self.saw_sdpa = False

    def transform_for_pre_decomposition(
        self, exported_program: ExportedProgram
    ) -> ExportedProgram:
        self.calls.append(self.name)
        self.saw_sdpa = any(
            node.target == torch.ops.aten.scaled_dot_product_attention.default
            for node in exported_program.graph.nodes
        )
        return exported_program

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        return PartitionResult(exported_program, {})


def test_partitioner_transforms_run_before_decomposition_in_order() -> None:
    inputs = tuple(torch.randn(1, 3, 4, 5) for _ in range(3))
    exported_program = export(_SDPA(), inputs, strict=True)
    calls: list[str] = []
    first = _RecordingPartitioner("first", calls)
    second = _RecordingPartitioner("second", calls)

    to_edge_transform_and_lower(
        exported_program,
        partitioner=[first, second],
    )

    assert calls == ["first", "second"]
    assert first.saw_sdpa
    assert second.saw_sdpa
