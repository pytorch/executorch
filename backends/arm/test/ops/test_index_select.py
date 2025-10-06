# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)


class IndexSelect(torch.nn.Module):
    aten_op = "torch.ops.aten.index_select.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_index_select_default"

    def forward(self, input_: torch.Tensor, dim, index_: torch.Tensor):
        return torch.index_select(input_, dim=dim, index=index_)


input_params = Tuple[torch.Tensor, int, torch.Tensor]


test_input: dict[input_params] = {
    "test_1": (
        torch.tensor(
            [[[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3]]],
            dtype=torch.float32,
        ),  # Shape: [N=1, K=4, C=3]
        1,
        torch.tensor(
            [1, 3], dtype=torch.int32
        ),  # Shape: [2] => Note TOSA requires [N=1, W=2]
    ),
    "test_2": (
        torch.tensor(
            [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3]],
            dtype=torch.float32,
        ),  # Shape: [K=4, C=3]
        0,
        torch.tensor(
            [1, 3], dtype=torch.int32
        ),  # Shape: [2] => Note TOSA requires [N=1, W=2]
    ),
    "test_3_mult_batches": (
        torch.randn(2, 4, 3),  # Batches > 1 not supported
        1,
        torch.tensor(
            [1, 3], dtype=torch.int32
        ),  # Shape: [2] => Note TOSA requires [N=1, W=2]
    ),
    "test_4_rand": (
        torch.randn(1, 4, 3),
        1,
        torch.tensor(
            [1, 3], dtype=torch.int32
        ),  # Shape: [2] => Note TOSA requires [N=1, W=2]
    ),
}


test_data = {
    "index_select_test_1": (IndexSelect(), test_input["test_1"]),
    "index_select_test_2": (IndexSelect(), test_input["test_2"]),
    "index_select_test_3": pytest.param(
        (IndexSelect(), test_input["test_3_mult_batches"]),
        marks=pytest.mark.xfail(
            reason="Rank3 weights with first dim larger than 1 is currently not supported"
        ),
    ),
    "index_select_test_4": (IndexSelect(), test_input["test_4_rand"]),
}


@pytest.mark.parametrize("test_data", list(test_data.values()))
def test_index_select_tosa_FP(test_data: input_params):
    op, test_input = test_data
    pipeline = TosaPipelineFP[input_params](
        op, test_input, op.aten_op, op.exir_op, use_to_edge_transform_and_lower=True
    )
    pipeline.run()


@pytest.mark.parametrize("test_data", list(test_data.values())[:-1])
def test_index_select_tosa_INT(test_data: input_params):
    op, test_input = test_data

    pipeline = TosaPipelineINT[input_params](
        op,
        test_input,
        op.aten_op,
        op.exir_op,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.run()


@pytest.mark.parametrize("test_data", list(test_data.values())[-1:])
def test_index_select_tosa_INT_rand(test_data: input_params):
    op, test_input = test_data

    pipeline = TosaPipelineINT[input_params](
        op,
        test_input,
        op.aten_op,
        op.exir_op,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.change_args(
        "run_method_and_compare_outputs", inputs=test_input, atol=0.9, rtol=0.2, qtol=1
    )
    pipeline.run()


@pytest.mark.parametrize("test_data", list(test_data.values())[-1:])
def test_index_select_u55_INT_not_delegated(test_data: input_params):
    op, test_input = test_data

    pipeline = OpNotSupportedPipeline[input_params](
        op,
        test_input,
        {op.exir_op: 1},
        quantize=True,
        u55_subset=True,
    )
    pipeline.run()


@pytest.mark.parametrize("test_data", list(test_data.values()))
@common.SkipIfNoModelConverter
def test_index_select_vgf_FP(test_data: input_params):
    op, inp = test_data
    pipeline = VgfPipeline[input_params](
        op,
        inp,
        op.aten_op,
        op.exir_op,
        tosa_version="TOSA-1.0+FP",
    )
    pipeline.run()


@pytest.mark.parametrize("test_data", list(test_data.values())[:-1])
@common.SkipIfNoModelConverter
def test_index_select_vgf_INT(test_data: input_params):
    op, inp = test_data
    pipeline = VgfPipeline[input_params](
        op,
        inp,
        op.aten_op,
        op.exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()


@pytest.mark.parametrize("test_data", list(test_data.values())[-1:])
@common.SkipIfNoModelConverter
def test_index_select_vgf_INT_rand(test_data: input_params):
    op, inp = test_data
    pipeline = VgfPipeline[input_params](
        op,
        inp,
        op.aten_op,
        op.exir_op,
        tosa_version="TOSA-1.0+INT",
    )
    pipeline.run()
