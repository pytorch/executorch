# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest
import torch
from executorch.backends.arm._passes import InsertInt32CastsAfterInt64PlaceholdersPass

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)


class Embedding(torch.nn.Module):

    aten_op = "torch.ops.aten.embedding.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_embedding_default"

    def forward(self, weights: torch.Tensor, indices: torch.Tensor):
        return torch.embedding(weights, indices)


class ExpandEmbedding(Embedding):
    example_inputs = (torch.randn(10, 3), torch.tensor([[1, 2, 3]], dtype=torch.int32))

    def forward(self, weights: torch.Tensor, indices: torch.Tensor):
        return torch.embedding(weights, indices.expand(2, 3))


input_params = Tuple[torch.Tensor, torch.Tensor]


test_input: dict[str, input_params] = {
    "test_1": (
        torch.randn(10, 3),
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
    ),
    "test_2": (
        torch.randn(10, 4),
        torch.tensor([[1, 4, 3], [4, 3, 2]], dtype=torch.int32),
    ),
    "test_3": (
        torch.randn(9, 3),
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64),
    ),
    "test_4": (
        torch.randn(11, 5),
        torch.randint(low=0, high=10, size=(4, 3), dtype=torch.int64),
    ),
    "test_5": (
        torch.randn(11, 5),
        torch.randint(low=0, high=10, size=(4, 3, 2), dtype=torch.int64),
    ),
    "test_6": (
        torch.randn(11, 5),
        torch.randint(low=0, high=10, size=(4, 3, 2, 5), dtype=torch.int64),
    ),
}
test_input_fp8: dict[str, tuple[input_params, str]] = {
    "test_fp8e4m3_int32_indices": (
        (
            torch.randn(10, 3, dtype=torch.float32).to(torch.float8_e4m3fn),
            torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
        ),
        "fp8e4m3",
    ),
    "test_fp8e5m2_int64_indices": (
        (
            torch.randn(11, 5, dtype=torch.float32).to(torch.float8_e5m2),
            torch.randint(low=0, high=10, size=(4, 3), dtype=torch.int64),
        ),
        "fp8e5m2",
    ),
}


@pytest.mark.skip(reason="MLETORCH-1274 Improve data type checks during partitioning")
@common.parametrize("test_input", test_input)
def test_embedding_tosa_FP(test_input: input_params):
    op = Embedding()
    pipeline = TosaPipelineFP[input_params](
        op,
        test_input,
        op.aten_op,
        op.exir_op,
        transform_passes=[InsertInt32CastsAfterInt64PlaceholdersPass()],
    )
    pipeline.run()


@common.parametrize("test_input", test_input)
def test_embedding_tosa_INT(test_input: input_params):
    op = Embedding()
    pipeline = TosaPipelineINT[input_params](
        op,
        test_input,
        op.aten_op,
        op.exir_op,
    )
    pipeline.pop_stage("check.aten")
    pipeline.pop_stage("check_count.exir")

    pipeline.run()


@common.parametrize("test_input", test_input_fp8)
def test_embedding_tosa_FP_fp8(test_input):
    inputs, tosa_extension = test_input
    op = Embedding()
    pipeline = TosaPipelineFP[input_params](
        op,
        inputs,
        op.aten_op,
        op.exir_op,
        transform_passes=[InsertInt32CastsAfterInt64PlaceholdersPass()],
        compare_tosa_ref_model_outputs=False,
        tosa_extensions=[tosa_extension],
    )
    pipeline.run()


def test_embedding_tosa_INT_expand():
    op = ExpandEmbedding()
    pipeline = TosaPipelineINT(
        op,
        ExpandEmbedding.example_inputs,
        ExpandEmbedding.aten_op,
        ExpandEmbedding.exir_op,
    )
    pipeline.pop_stage("check.aten")
    pipeline.pop_stage("check_count.exir")

    pipeline.run()


@pytest.mark.skip("reason=MLETORCH-1274 Improve data type checks during partitioning")
@common.parametrize("test_input", test_input)
@common.SkipIfNoModelConverter
def test_embedding_vgf_no_quant(test_input: input_params):
    op = Embedding()
    pipeline = VgfPipeline[input_params](
        op,
        test_input,
        op.aten_op,
        op.exir_op,
        transform_passes=[InsertInt32CastsAfterInt64PlaceholdersPass()],
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_input", test_input)
@common.SkipIfNoModelConverter
def test_embedding_vgf_quant(test_input: input_params):
    op = Embedding()
    pipeline = VgfPipeline[input_params](
        op,
        test_input,
        op.aten_op,
        op.exir_op,
        quantize=True,
    )
    pipeline.pop_stage("check.aten")
    pipeline.pop_stage("check_count.exir")

    pipeline.run()
