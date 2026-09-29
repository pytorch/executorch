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

input_t = Tuple[int, torch.Tensor, torch.LongTensor, torch.Tensor]  # dim, x, index, y


class IndexCopyModule(torch.nn.Module):
    base_test_data = {
        "rand_1d": lambda: (
            0,
            torch.rand((6,), dtype=torch.float32),
            torch.LongTensor([0, 2, 5]),
            torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32),
        ),
        "rand_3d": lambda: (
            0,
            torch.rand((4, 2, 3), dtype=torch.float32),
            torch.LongTensor([0, 3]),
            torch.ones((2, 2, 3), dtype=torch.float32),
        ),
        "rand_3d_dim_1": lambda: (
            1,
            torch.rand((4, 2, 3), dtype=torch.float32),
            torch.LongTensor([0, 1]),
            torch.ones((4, 2, 3), dtype=torch.float32),
        ),
        "rand_3d_dim_2": lambda: (
            2,
            torch.rand((4, 2, 3), dtype=torch.float32),
            torch.LongTensor([0]),
            torch.ones((4, 2, 1), dtype=torch.float32),
        ),
        "rand_single_index": lambda: (
            0,
            torch.rand((4, 5), dtype=torch.float32),
            torch.LongTensor([0]),
            torch.zeros((1, 5), dtype=torch.float32),
        ),
        "rand_single_index_not_zero": lambda: (
            0,
            torch.rand((4, 5), dtype=torch.float32),
            torch.LongTensor([2]),
            torch.zeros((1, 5), dtype=torch.float32),
        ),
        "rand_all_rows": lambda: (
            0,
            torch.rand((3, 4), dtype=torch.float32),
            torch.LongTensor([0, 1, 2]),
            torch.ones((3, 4), dtype=torch.float32),
        ),
    }

    test_data = {
        f"{name}_{variant}": (
            lambda test_case=test_case, inplace=inplace: (test_case(), inplace)
        )
        for name, test_case in base_test_data.items()
        for variant, inplace in (
            ("out_of_place", False),
            ("in_place", True),
        )
    }
    test_data_fp8 = {
        "rand_single_index_fp8e4m3_out_of_place": (
            lambda: (
                (
                    0,
                    torch.rand((4, 5), dtype=torch.float32).to(torch.float8_e4m3fn),
                    torch.LongTensor([0]),
                    torch.zeros((1, 5), dtype=torch.float32).to(torch.float8_e4m3fn),
                ),
                False,
                "fp8e4m3",
            )
        ),
        "rand_3d_dim_1_fp8e5m2_in_place": (
            lambda: (
                (
                    1,
                    torch.rand((4, 2, 3), dtype=torch.float32).to(torch.float8_e5m2),
                    torch.LongTensor([0, 1]),
                    torch.ones((4, 2, 3), dtype=torch.float32).to(torch.float8_e5m2),
                ),
                True,
                "fp8e5m2",
            )
        ),
    }

    aten_ops = {
        False: ["torch.ops.aten.index_put.default"],
        True: ["torch.ops.aten.index_put_.default"],
    }
    exir_op = "executorch_exir_dialects_edge__ops_aten_index_put_default"

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(
        self, dim: int, x: torch.Tensor, index: torch.LongTensor, y: torch.Tensor
    ):
        if self.inplace:
            return x.index_copy_(dim, index, y)
        return x.index_copy(dim, index, y)


@common.parametrize("test_data", IndexCopyModule.test_data)
def test_index_copy_tosa_FP(test_data):
    inputs, inplace = test_data()
    module = IndexCopyModule(inplace=inplace)
    pipeline = TosaPipelineFP(
        module=module,
        test_data=inputs,
        aten_op=[],
        transform_passes=[InsertInt32CastsAfterInt64PlaceholdersPass()],
    )
    pipeline.run()


@common.parametrize("test_data", IndexCopyModule.test_data_fp8)
def test_index_copy_tosa_FP_fp8(test_data):
    inputs, inplace, tosa_extension = test_data()
    module = IndexCopyModule(inplace=inplace)
    pipeline = TosaPipelineFP(
        module=module,
        test_data=inputs,
        aten_op=[],
        compare_tosa_ref_model_outputs=False,
        transform_passes=[InsertInt32CastsAfterInt64PlaceholdersPass()],
        tosa_extensions=[tosa_extension],
    )
    pipeline.run()


@common.parametrize("test_data", IndexCopyModule.test_data)
def test_index_copy_tosa_INT(test_data):
    inputs, inplace = test_data()
    module = IndexCopyModule(inplace=inplace)
    pipeline = TosaPipelineINT(
        module=module,
        test_data=inputs,
        aten_op=IndexCopyModule.aten_ops[inplace],
    )
    pipeline.run()


@common.parametrize("test_data", IndexCopyModule.test_data)
def test_index_copy_u55_INT(test_data):
    inputs, inplace = test_data()
    # SCATTER (index_put) is not supported on U55
    pipeline = OpNotSupportedPipeline[input_t](
        IndexCopyModule(inplace=inplace),
        inputs,
        {IndexCopyModule.exir_op: 1},
        quantize=True,
        u55_subset=True,
        n_expected_delegates=0,
    )
    pipeline.run()


@common.parametrize("test_data", IndexCopyModule.test_data)
@common.XfailIfNoCorstone320
def test_index_copy_u85_INT(test_data):
    inputs, inplace = test_data()
    pipeline = EthosU85PipelineINT[input_t](
        IndexCopyModule(inplace=inplace),
        inputs,
        aten_ops=IndexCopyModule.aten_ops[inplace],
    )
    # int64 index inputs need to be cast to int32; _to_dim_order_copy is not delegated
    pipeline.tester.use_portable_ops = True
    pipeline.run()


@common.parametrize("test_data", IndexCopyModule.test_data)
@common.SkipIfNoModelConverter
def test_index_copy_vgf_no_quant(test_data):
    inputs, inplace = test_data()
    pipeline = VgfPipeline[input_t](
        IndexCopyModule(inplace=inplace),
        inputs,
        aten_op=[],
        transform_passes=[InsertInt32CastsAfterInt64PlaceholdersPass()],
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", IndexCopyModule.test_data)
@common.SkipIfNoModelConverter
def test_index_copy_vgf_quant(test_data):
    inputs, inplace = test_data()
    pipeline = VgfPipeline[input_t](
        IndexCopyModule(inplace=inplace),
        inputs,
        aten_op=IndexCopyModule.aten_ops[inplace],
        quantize=True,
        tosa_spec="TOSA-1.0+INT",
    )
    pipeline.run()
