# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm._passes import InsertInt32CastsAfterInt64PlaceholdersPass

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)


class Gather(torch.nn.Module):
    aten_op = "torch.ops.aten.gather.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_gather_default"

    def forward(self, input_: torch.Tensor, dim_, index_: torch.Tensor):
        return torch.gather(input_, dim=dim_, index=index_)


input_params = Tuple[torch.Tensor, int, torch.Tensor]


# FP profile: only float inputs.
test_data_fp: dict[str, input_params] = {
    "test_fp32": (
        torch.tensor(
            [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3]],
            dtype=torch.float32,
        ),  # Shape: [N=4, K=3]
        1,
        torch.tensor(
            [[1, 0], [1, 2], [0, 2], [2, 0]],
            dtype=torch.int64,
        ),  # Shape: [N=4, W=2]
    ),
}


# INT profile: integer inputs + bool (bool is supported via casts in
# CanonicalizeGatherPass: bool -> int8 -> bool).
test_data_int: dict[str, input_params] = {
    "test_int32": (
        torch.tensor(
            [[1, 2, 3], [11, 12, 13], [21, 22, 23], [31, 32, 33]],
            dtype=torch.int32,
        ),  # Shape: [N=4, K=3]
        1,
        torch.tensor(
            [[1, 1], [2, 1], [1, 2], [2, 0]],
            dtype=torch.int64,
        ),  # Shape: [N=4, W=2]
    ),
    "test_int8": (
        torch.randint(5, size=(5, 10), dtype=torch.int8),  # Shape: [N=5, K=10]
        1,
        torch.tensor(
            [[3, 2, 1], [2, 4, 0], [1, 0, 2], [2, 3, 1], [4, 1, 2]],
            dtype=torch.int64,
        ),  # Shape: [N=5, W=3]
    ),
    "test_bool": (
        torch.tensor(
            [
                [True, False, True],
                [False, True, True],
                [False, True, False],
                [True, True, False],
            ],
            dtype=torch.bool,
        ),  # Shape: [N=4, K=3]
        1,
        torch.tensor(
            [[1, 0], [1, 2], [0, 2], [2, 0]],
            dtype=torch.int64,
        ),  # Shape: [N=4, W=2]
    ),
}


@common.parametrize("test_data", test_data_fp)
def test_gather_tosa_FP(test_data: input_params):
    pipeline = TosaPipelineFP[input_params](
        Gather(),
        test_data,
        aten_op=Gather.aten_op,
        exir_op=Gather.exir_op,
        transform_passes=[
            InsertInt32CastsAfterInt64PlaceholdersPass(),
        ],  # int64 index are not currently supported and need to be cast to int32
    )
    pipeline.run()


@common.parametrize("test_data", test_data_int | test_data_fp)
def test_gather_tosa_INT(test_data: input_params):
    pipeline = TosaPipelineINT[input_params](
        Gather(),
        test_data,
        aten_op=Gather.aten_op,
        exir_op=Gather.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_int | test_data_fp)
def test_gather_u55_INT(test_data: input_params):
    # Gather op is not supported on U55
    pipeline = OpNotSupportedPipeline[input_params](
        Gather(),
        test_data,
        {Gather.exir_op: 1},
        quantize=True,
        u55_subset=True,
        n_expected_delegates=0,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_int | test_data_fp)
def test_gather_u85_INT(test_data: input_params):
    pipeline = EthosU85PipelineINT[input_params](
        Gather(),
        test_data,
        aten_ops=Gather.aten_op,
        exir_ops=Gather.exir_op,
    )
    # U85: keep _to_dim_order_copy portable for int64->int32 index casts (not delegatable).
    pipeline.tester.use_portable_ops = True
    pipeline.run()


@common.parametrize("test_data", test_data_fp | test_data_int)
@common.SkipIfNoModelConverter
def test_gather_vgf_no_quant(test_data: input_params):
    pipeline = VgfPipeline[input_params](
        Gather(),
        test_data,
        aten_op=Gather.aten_op,
        exir_op=Gather.exir_op,
        quantize=False,
        transform_passes=[
            InsertInt32CastsAfterInt64PlaceholdersPass(),
        ],  # int64 index are not currently supported and need to be cast to int32
    )
    pipeline.run()


@common.parametrize("test_data", test_data_fp | test_data_int)
@common.SkipIfNoModelConverter
def test_gather_vgf_quant(test_data: input_params):
    pipeline = VgfPipeline[input_params](
        Gather(),
        test_data,
        aten_op=Gather.aten_op,
        exir_op=Gather.exir_op,
        quantize=True,
    )
    pipeline.run()
