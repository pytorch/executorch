# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm._passes import CastZeroDimInt64OperandsPass
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as edge_ops


class Op(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.op(*args)


_BYTES = torch.tensor([1, -2, 3, 4], dtype=torch.int8)
_WIDE_SCALAR = torch.tensor(2**40 + 7)

test_cases = {
    "where_int8": (
        torch.where,
        (_BYTES > 0, _BYTES, _WIDE_SCALAR),
        ["int64"],
        [torch.int8],
    ),
    "add_int32": (torch.add, (_BYTES.int(), _WIDE_SCALAR), ["int64"], [torch.int32]),
    "index_select": (
        lambda x, index: torch.index_select(x, 0, index),
        (_BYTES, torch.tensor(3)),
        ["int64"],
        [],
    ),
    "or_without_int64": (torch.bitwise_or, (_BYTES, _WIDE_SCALAR), [], []),
}


@pytest.mark.parametrize(
    ("op", "inputs", "extensions", "dtypes"), test_cases.values(), ids=test_cases.keys()
)
def test_cast_zero_dim_int64_operands(op, inputs, extensions, dtypes) -> None:
    pipeline = PassPipeline(
        Op(op),
        inputs,
        tosa_version="1.1",
        tosa_extensions=["INT", *extensions],
        pass_list=[CastZeroDimInt64OperandsPass],
    )
    pipeline.run()

    graph_module = (
        pipeline.tester.get_artifact(StageType.RUN_PASSES)
        .exported_program()
        .graph_module
    )
    cast_op = edge_ops.edge.dim_order_ops._to_dim_order_copy.default
    casts = [node for node in graph_module.graph.nodes if node.target == cast_op]
    assert [node.kwargs["dtype"] for node in casts] == dtypes
