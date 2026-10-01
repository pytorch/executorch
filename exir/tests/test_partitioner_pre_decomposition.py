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


class _MarkRecordingPartitioner(Partitioner):
    def __init__(self) -> None:
        super().__init__()
        self.marked: set[str] = set()

    def transform_for_pre_decomposition(
        self, exported_program: ExportedProgram
    ) -> ExportedProgram:
        parameters = exported_program.graph_signature.inputs_to_parameters
        self.marked = {
            parameters[node.name]
            for node in exported_program.graph.find_nodes(op="placeholder")
            if node.meta.get("shared_across_methods")
        }
        return exported_program

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        return PartitionResult(exported_program, {})


class _TwoMethods(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Parameter(torch.ones(4))
        self.only_a = torch.nn.Parameter(torch.ones(4))
        self.only_b = torch.nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.shared + self.only_a

    def method_b(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.shared + self.only_b


def test_constants_read_by_several_methods_are_marked_for_the_transforms() -> None:
    """
    A transform sees one method at a time, so EXIR marks the constants that
    another method of the program also reads. A parameter a method lifts
    but never uses, as a non-strict export of one module lifts them all, is
    not a read; a program with one method has nothing to mark.
    """
    model = _TwoMethods()
    x = torch.ones(4)
    method_a = export(model, (x,))
    model.forward = model.method_b
    method_b = export(model, (x,))
    recorders = {"a": _MarkRecordingPartitioner(), "b": _MarkRecordingPartitioner()}

    to_edge_transform_and_lower(
        {"a": method_a, "b": method_b},
        partitioner={name: [recorder] for name, recorder in recorders.items()},
    )

    assert recorders["a"].marked == {"shared"}
    assert recorders["b"].marked == {"shared"}

    alone = _MarkRecordingPartitioner()
    to_edge_transform_and_lower(export(_TwoMethods(), (x,)), partitioner=[alone])
    assert alone.marked == set()
