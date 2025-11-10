# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools

from typing import Any, Callable, Tuple

import torch
from executorch.backends.arm.quantizer import is_annotated
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT
from executorch.backends.test.harness.stages import StageType

from torch.fx.passes.utils.source_matcher_utils import get_source_partitions


input_t1 = Tuple[torch.Tensor]  # Input x


class SingleOpModel(torch.nn.Module):
    def __init__(
        self,
        op: Callable[..., torch.Tensor],
        example_input: Tuple[Any, ...],
        **op_kwargs: Any,
    ) -> None:
        super().__init__()
        self.op: Callable[..., torch.Tensor] = op
        self._example_input: Tuple[Any, ...] = example_input
        self.op_kwargs: dict[str, Any] = dict(op_kwargs)

    def forward(self, x: Any) -> torch.Tensor:
        return self.op(x, **self.op_kwargs)

    def example_inputs(self) -> Tuple[Any, ...]:
        return self._example_input


def check_annotation(model: SingleOpModel) -> None:
    pipeline = TosaPipelineINT[input_t1](model, model.example_inputs(), [], [])
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()

    artifact = pipeline.tester.get_artifact(StageType.QUANTIZE)

    partitions = get_source_partitions(artifact.graph, [model.op])
    partitions = list(itertools.chain.from_iterable(partitions.values()))

    assert len(partitions) == 1
    partition = partitions[0]
    assert all(is_annotated(node) for node in partition.nodes)


def test_squeeze():
    check_annotation(SingleOpModel(torch.squeeze, (torch.rand(8, 8, 1),)))
    check_annotation(SingleOpModel(torch.squeeze_copy, (torch.rand(8, 8, 1),)))


def test_unsqueeze():
    check_annotation(SingleOpModel(torch.unsqueeze, (torch.rand(8, 8),), dim=0))
    check_annotation(SingleOpModel(torch.unsqueeze_copy, (torch.rand(8, 8),), dim=0))


def test_reshape():
    check_annotation(
        SingleOpModel(torch.reshape, (torch.randn(8, 8),), shape=(64,)),
    )


def test_view():
    check_annotation(
        SingleOpModel(torch.view_copy, (torch.randn(4, 4),), size=(2, 8)),
    )


def test_slice():
    check_annotation(
        SingleOpModel(torch.slice_copy, (torch.randn(3, 4),)),
    )


def test_transpose():
    check_annotation(
        SingleOpModel(torch.transpose, (torch.randn(2, 3),), dim0=0, dim1=1),
    )
    check_annotation(
        SingleOpModel(torch.transpose_copy, (torch.randn(2, 3),), dim0=0, dim1=1),
    )


def test_tile():
    check_annotation(
        SingleOpModel(torch.tile, (torch.randn(4, 4),), dims=(2,)),
    )


def test_flip():
    check_annotation(
        SingleOpModel(torch.flip, (torch.randn(2, 4),), dims=(0, 1)),
    )


def test_concat():
    check_annotation(
        SingleOpModel(
            torch.concatenate, ((torch.randn(2, 3), torch.randn(2, 3)),), dim=0
        ),
    )
