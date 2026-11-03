# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import pytest

import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.backend.partitioner import Partitioner, PartitionResult
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.pass_manager import PassType
from torch.export import export, ExportedProgram


class _Add(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


class _RecordingPass(ExportPass):
    def __init__(self, name: str, calls: List[str]) -> None:
        super().__init__()
        self.name = name
        self.calls = calls

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        self.calls.append(self.name)
        return PassResult(graph_module, False)


class _ContributingPartitioner(Partitioner):
    def __init__(self, names: List[str], calls: List[str]) -> None:
        super().__init__()
        self.passes = [_RecordingPass(name, calls) for name in names]
        self.saw_program = False

    def transform_passes(self, exported_program: ExportedProgram) -> List[PassType]:
        self.saw_program = isinstance(exported_program, ExportedProgram)
        return list(self.passes)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        return PartitionResult(exported_program, {})


class _NoopPartitioner(Partitioner):
    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        return PartitionResult(exported_program, {})


def _lower(partitioners, transform_passes=None):
    exported_program = export(_Add(), (torch.randn(2, 3),), strict=True)
    return to_edge_transform_and_lower(
        exported_program,
        transform_passes=transform_passes,
        partitioner=partitioners,
    )


def test_partitioner_transform_passes_run() -> None:
    calls: List[str] = []
    partitioner = _ContributingPartitioner(["a"], calls)

    _lower([partitioner])

    assert calls == ["a"]
    assert partitioner.saw_program


def test_partitioner_transform_passes_default_is_empty() -> None:
    # A partitioner that does not override the hook must not change behavior.
    _lower([_NoopPartitioner()])


def test_partitioner_transform_passes_concatenate_in_order() -> None:
    calls: List[str] = []
    first = _ContributingPartitioner(["first"], calls)
    second = _ContributingPartitioner(["second"], calls)

    _lower([first, second])

    assert calls == ["first", "second"]


def test_caller_transform_passes_run_before_partitioner_passes() -> None:
    calls: List[str] = []
    partitioner = _ContributingPartitioner(["from_partitioner"], calls)

    _lower([partitioner], transform_passes=[_RecordingPass("from_caller", calls)])

    assert calls == ["from_caller", "from_partitioner"]


def test_pass_manager_argument_with_contributed_passes_is_rejected() -> None:
    from torch.fx.passes.infra.pass_manager import PassManager

    calls: List[str] = []
    partitioner = _ContributingPartitioner(["a"], calls)

    with pytest.raises(ValueError, match="cannot be combined"):
        _lower([partitioner], transform_passes=PassManager([]))
