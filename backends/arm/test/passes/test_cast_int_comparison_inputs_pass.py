# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from collections.abc import Callable

import pytest
import torch
from executorch.backends.arm._passes import CastIntComparisonInputsPass
from executorch.backends.arm.test.tester.test_pipeline import (
    PassPipeline,
    TosaPipelineFP,
)
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as edge_ops


class Comparison(torch.nn.Module):
    def __init__(self, op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        super().__init__()
        self.op = op

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.op(x, y)


class ScalarComparison(torch.nn.Module):
    def __init__(self, op: Callable[[torch.Tensor, int], torch.Tensor]):
        super().__init__()
        self.op = op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x, 0)


comparison_ops = {
    "eq": operator.eq,
    "ne": operator.ne,
    "ge": operator.ge,
    "gt": operator.gt,
    "le": operator.le,
    "lt": operator.lt,
}
aten_ops = {
    "eq": "torch.ops.aten.eq.Tensor",
    "ne": "torch.ops.aten.ne.Tensor",
    "ge": "torch.ops.aten.ge.Tensor",
    "gt": "torch.ops.aten.gt.Tensor",
    "le": "torch.ops.aten.le.Tensor",
    "lt": "torch.ops.aten.lt.Tensor",
}
aten_scalar_ops = {
    name: target.replace("Tensor", "Scalar") for name, target in aten_ops.items()
}
exir_ops = {
    name: f"executorch_exir_dialects_edge__ops_aten_{name}_Tensor"
    for name in comparison_ops
}
exir_scalar_ops = {
    name: target.replace("Tensor", "Scalar") for name, target in exir_ops.items()
}


def comparison_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    limits = torch.iinfo(dtype)
    return (
        torch.tensor(
            [limits.min, limits.min + 1, limits.max - 1, limits.max], dtype=dtype
        ),
        torch.tensor(
            [limits.min + 1, limits.min, limits.max, limits.max - 1], dtype=dtype
        ),
    )


@pytest.mark.parametrize("op", comparison_ops.values(), ids=comparison_ops.keys())
@pytest.mark.parametrize(
    ("dtypes", "expected_dtype"),
    (
        ((torch.int8, torch.int8), torch.float16),
        ((torch.int16, torch.int16), torch.float32),
        ((torch.int8, torch.int16), torch.float32),
    ),
)
def test_cast_int_comparison_inputs(op, dtypes, expected_dtype) -> None:
    inputs = (
        comparison_inputs(dtypes[0])[0],
        comparison_inputs(dtypes[1])[1],
    )
    pipeline = PassPipeline(
        Comparison(op),
        inputs,
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 2
        },
        pass_list=[CastIntComparisonInputsPass],
    )
    pipeline.run()

    graph_module = (
        pipeline.tester.get_artifact(StageType.RUN_PASSES)
        .exported_program()
        .graph_module
    )
    cast_op = edge_ops.edge.dim_order_ops._to_dim_order_copy.default
    cast_nodes = [node for node in graph_module.graph.nodes if node.target == cast_op]
    assert len(cast_nodes) == 2
    assert all(node.kwargs["dtype"] == expected_dtype for node in cast_nodes)


def test_cast_int_comparison_inputs_keeps_int32() -> None:
    inputs = (
        torch.tensor([2**24, 2**24 + 1], dtype=torch.int32),
        torch.tensor([2**24 + 1, 2**24], dtype=torch.int32),
    )
    pipeline = PassPipeline(
        Comparison(operator.eq),
        inputs,
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default"
        ],
        pass_list=[CastIntComparisonInputsPass],
    )
    pipeline.run()


@pytest.mark.parametrize("name", comparison_ops)
@pytest.mark.parametrize("dtype", (torch.int8, torch.int16))
def test_int_comparison_tosa_fp(name, dtype) -> None:
    inputs = comparison_inputs(dtype)
    pipeline = TosaPipelineFP(
        Comparison(comparison_ops[name]),
        inputs,
        aten_ops[name],
        exir_ops[name],
    )
    pipeline.run()


@pytest.mark.parametrize("name", comparison_ops)
@pytest.mark.parametrize("dtype", (torch.int8, torch.int16))
def test_int_scalar_comparison_tosa_fp(name, dtype) -> None:
    inputs = (comparison_inputs(dtype)[0],)
    pipeline = TosaPipelineFP(
        ScalarComparison(comparison_ops[name]),
        inputs,
        aten_scalar_ops[name],
        exir_scalar_ops[name],
    )
    pipeline.run()
