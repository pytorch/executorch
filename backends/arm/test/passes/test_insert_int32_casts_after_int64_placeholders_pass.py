# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from executorch.backends.arm._passes import InsertInt32CastsAfterInt64PlaceholdersPass

from executorch.backends.arm.test.tester.test_pipeline import (
    PassPipeline,
    TosaPipelineINT,
)
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as edge_ops

input_t = Tuple[torch.Tensor, torch.Tensor]  # weights, indices
input_t3 = Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]


class Int64InputModel(torch.nn.Module):
    def forward(self, weights: torch.Tensor, indices: torch.Tensor):
        return torch.embedding(weights, indices)

    def get_inputs(self) -> input_t:
        return (
            torch.randn(9, 3),
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64),
        )


def test_insert_int32_casts_after_int64_placeholders_tosa_FP():
    module = Int64InputModel()
    op_checks_before = {
        "executorch_exir_dialects_edge__ops_aten_embedding_default": 1,
    }
    op_checks_after = {
        "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1,
        "executorch_exir_dialects_edge__ops_aten_embedding_default": 1,
    }

    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        ops_before_pass=op_checks_before,
        ops_after_pass=op_checks_after,
        pass_list=[InsertInt32CastsAfterInt64PlaceholdersPass],
    )
    pipeline.pop_stage(-1)  # Do not compare output
    pipeline.run()


class UpcastToInt64ForIndexCopyInplaceModel(torch.nn.Module):
    aten_ops = [
        "torch.ops.dim_order_ops._to_dim_order_copy.default",
        "torch.ops.aten.index_put_.default",
    ]

    def forward(self, x: torch.Tensor, index: torch.LongTensor, y: torch.Tensor):
        return x.index_copy_(0, index, y)

    def get_inputs(self) -> input_t3:
        return (
            torch.zeros(5, 3),
            torch.LongTensor([0, 4, 2]),
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float),
        )


def test_insert_int32_casts_after_int64_placeholders_tosa_INT_upcast_for_index_copy_inplace():
    module = UpcastToInt64ForIndexCopyInplaceModel()

    # In TOSA+INT index_copy_ decomposes to index_put_
    # There should also be cast from int64 to int32
    pipeline = TosaPipelineINT[input_t3](
        module,
        module.get_inputs(),
        aten_op=UpcastToInt64ForIndexCopyInplaceModel.aten_ops,
    )
    pipeline.run()


class UpcastToInt64ForIndexCopyModel(torch.nn.Module):
    aten_ops = [
        "torch.ops.dim_order_ops._to_dim_order_copy.default",
        "torch.ops.aten.index_put.default",
    ]

    def forward(self, x: torch.Tensor, index: torch.LongTensor, y: torch.Tensor):
        return x.index_copy(0, index, y)

    def get_inputs(self) -> input_t3:
        return (
            torch.zeros(5, 3),
            torch.LongTensor([0, 4, 2]),
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float),
        )


def test_insert_int32_casts_after_int64_placeholders_tosa_INT_upcast_for_index_copy():
    module = UpcastToInt64ForIndexCopyModel()

    # In TOSA+INT index_copy decomposes to index_put
    # There should also be cast from int64 to int32
    pipeline = TosaPipelineINT[input_t3](
        module,
        module.get_inputs(),
        aten_op=UpcastToInt64ForIndexCopyModel.aten_ops,
    )
    pipeline.run()


class Int64FloatModel(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.op(x, y)


def reciprocal_add(x: torch.Tensor, y: torch.Tensor):
    return torch.reciprocal(x) + y


int64_float_test_cases = {
    "sub_tosa_FP": (torch.sub, False, [torch.float32]),
    "sub_tosa_INT": (torch.sub, True, [torch.int32, torch.float32]),
    "gt_tosa_FP": (torch.gt, False, [torch.float32]),
    "gt_tosa_INT": (torch.gt, True, [torch.int32]),
    "reciprocal_tosa_FP": (reciprocal_add, False, [torch.float32]),
    "reciprocal_tosa_INT": (reciprocal_add, True, []),
}


@pytest.mark.parametrize(
    ("op", "quantize", "cast_dtypes"),
    int64_float_test_cases.values(),
    ids=int64_float_test_cases.keys(),
)
def test_insert_int32_casts_after_int64_placeholders_int64_float_operand(
    op, quantize, cast_dtypes
):
    pipeline = PassPipeline[input_t](
        Int64FloatModel(op),
        (
            torch.tensor([1, -2, 3, 4], dtype=torch.int64),
            torch.tensor([0.5, 1.5, -2.5, 3.5]),
        ),
        quantize=quantize,
        tosa_version="1.1",
        tosa_extensions=["INT", "int64"],
        pass_list=[InsertInt32CastsAfterInt64PlaceholdersPass],
    )
    pipeline.pop_stage(-1)
    pipeline.run()

    graph_module = (
        pipeline.tester.get_artifact(StageType.RUN_PASSES)
        .exported_program()
        .graph_module
    )
    cast_op = edge_ops.edge.dim_order_ops._to_dim_order_copy.default
    casts = [node for node in graph_module.graph.nodes if node.target == cast_op]
    assert [node.kwargs["dtype"] for node in casts] == cast_dtypes
