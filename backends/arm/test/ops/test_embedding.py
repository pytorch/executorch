# Copyright 2025 Arm Limited and/or its affiliates.
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


input_params = Tuple[torch.Tensor, torch.Tensor, torch.dtype]


test_input: dict[input_params] = {
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


@pytest.mark.skip(reason="MLETORCH-1274 Improve data type checks during partitioning")
@common.parametrize("test_input", test_input)
def test_embedding_tosa_FP(test_input: input_params):
    op = Embedding()
    pipeline = TosaPipelineFP[input_params](
        op,
        test_input,
        op.aten_op,
        op.exir_op,
        use_to_edge_transform_and_lower=True,
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
        use_to_edge_transform_and_lower=True,
    )
    pipeline.pop_stage("check.aten")
    pipeline.pop_stage("check_count.exir")

    pipeline.run()


@pytest.mark.skip("reason=MLETORCH-1274 Improve data type checks during partitioning")
@common.parametrize("test_input", test_input)
@common.SkipIfNoModelConverter
def test_embedding_vgf_FP(test_input: input_params):
    op = Embedding()
    pipeline = VgfPipeline[input_params](
        op,
        test_input,
        op.aten_op,
        op.exir_op,
        tosa_version="TOSA-1.0+FP",
        use_to_edge_transform_and_lower=True,
        transform_passes=[InsertInt32CastsAfterInt64PlaceholdersPass()],
    )
    pipeline.run()


@common.parametrize("test_input", test_input)
@common.SkipIfNoModelConverter
def test_embedding_vgf_INT(test_input: input_params):
    op = Embedding()
    pipeline = VgfPipeline[input_params](
        op,
        test_input,
        op.aten_op,
        op.exir_op,
        tosa_version="TOSA-1.0+INT",
        use_to_edge_transform_and_lower=True,
    )
    pipeline.pop_stage("check.aten")
    pipeline.pop_stage("check_count.exir")

    pipeline.run()
