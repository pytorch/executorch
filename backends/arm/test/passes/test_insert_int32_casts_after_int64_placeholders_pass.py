# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes import InsertInt32CastsAfterInt64PlaceholdersPass

from executorch.backends.arm.test.tester.test_pipeline import (
    PassPipeline,
    TosaPipelineINT,
)

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


def test_int64_model_tosa_FP():
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
    aten_op = "torch.ops.aten.index_copy_.default"

    def forward(self, x: torch.Tensor, index: torch.LongTensor, y: torch.Tensor):
        return x.index_copy_(0, index, y)

    def get_inputs(self) -> input_t3:
        return (
            torch.zeros(5, 3),
            torch.LongTensor([0, 4, 2]),
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float),
        )


def test_upcast_to_int64_for_index_copy_inplace_tosa_INT():
    module = UpcastToInt64ForIndexCopyInplaceModel()
    pipeline = TosaPipelineINT[input_t3](
        module,
        module.get_inputs(),
        aten_op=module.aten_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.change_args(
        "check_count.exir",
        {
            "torch.ops.higher_order.executorch_call_delegate": 0,
        },
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


class UpcastToInt64ForIndexCopyModel(torch.nn.Module):
    aten_op = "torch.ops.aten.index_copy.default"

    def forward(self, x: torch.Tensor, index: torch.LongTensor, y: torch.Tensor):
        return x.index_copy(0, index, y)

    def get_inputs(self) -> input_t3:
        return (
            torch.zeros(5, 3),
            torch.LongTensor([0, 4, 2]),
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float),
        )


def test_upcast_to_int64_for_index_copy_tosa_INT():
    module = UpcastToInt64ForIndexCopyModel()
    pipeline = TosaPipelineINT[input_t3](
        module,
        module.get_inputs(),
        aten_op=module.aten_op,
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.change_args(
        "check_count.exir",
        {
            "torch.ops.higher_order.executorch_call_delegate": 0,
        },
    )
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()
