# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm._passes import RewriteInt64CastViaInt32Pass
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as edge_ops


class Cast(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.dtype)


test_cases = {
    "int8_to_int64": (torch.int8, torch.int64, ["int64"], [torch.int32, torch.int64]),
    "int64_to_int16": (torch.int64, torch.int16, ["int64"], [torch.int32, torch.int16]),
    "int32_to_int64": (torch.int32, torch.int64, ["int64"], [torch.int64]),
    "int8_to_int64_without_int64": (torch.int8, torch.int64, [], [torch.int64]),
}


@pytest.mark.parametrize(
    ("source", "target", "extensions", "dtypes"),
    test_cases.values(),
    ids=test_cases.keys(),
)
def test_rewrite_int64_cast_via_int32(source, target, extensions, dtypes) -> None:
    pipeline = PassPipeline(
        Cast(target),
        (torch.tensor([1, -2, 3, 4], dtype=source),),
        tosa_version="1.1",
        tosa_extensions=["INT", *extensions],
        pass_list=[RewriteInt64CastViaInt32Pass],
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
