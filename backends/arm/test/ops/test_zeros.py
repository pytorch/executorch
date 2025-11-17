# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

input_t = tuple[torch.Tensor]
test_data_t = tuple[int, torch.dtype]


class ZerosAdd(torch.nn.Module):
    aten_op: str = "torch.ops.aten.zeros.default"

    def __init__(self, n: int, dtype: torch.dtype):
        super().__init__()
        self.args = (n,)
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*self.args, dtype=self.dtype) + x

    test_data: dict[str, test_data_t] = {
        "10x3x3": (lambda: (torch.randn(10, 3, 3),), (3, torch.float32)),
        "10x1": (lambda: (torch.randn(10, 1),), (10, torch.float32)),
        "int32_int32": (
            lambda: (torch.randint(0, 10, [10], dtype=torch.int32),),
            (10, torch.int32),
        ),
    }

    # Mixed dtypes - the zeros op is delegated, but it leads to a non-delegated add op.
    test_data_mixed_dtypes: dict[str, test_data_t] = {
        "fp32_int64": (lambda: (torch.randn(10),), (10, torch.int64)),
        "fp32_int32": (lambda: (torch.randn(10),), (10, torch.int32)),
    }


@common.parametrize("test_data", ZerosAdd.test_data | ZerosAdd.test_data_mixed_dtypes)
def test_zeros_tosa_FP(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineFP[input_t](
        ZerosAdd(*init_data),
        input_data(),
        ZerosAdd.aten_op,
    )
    pipeline.run()


@common.parametrize("test_data", ZerosAdd.test_data | ZerosAdd.test_data_mixed_dtypes)
def test_zeros_tosa_INT(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = TosaPipelineINT[input_t](
        ZerosAdd(*init_data),
        input_data(),
        ZerosAdd.aten_op,
    )
    # Pop the quantization check stage if it exists as no
    # quantization nodes will be present for int + fp inputs.
    if pipeline.has_stage("check.quant_nodes"):
        pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", ZerosAdd.test_data)
@common.XfailIfNoCorstone300
def test_zeros_u55_INT(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = EthosU55PipelineINT[input_t](
        ZerosAdd(*init_data),
        input_data(),
        ZerosAdd.aten_op,
        use_to_edge_transform_and_lower=True,
    )
    # Pop the quantization check stage if it exists as no
    # quantization nodes will be present for int + fp inputs.
    if pipeline.has_stage("check.quant_nodes"):
        pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize("test_data", ZerosAdd.test_data)
@common.XfailIfNoCorstone320
def test_zeros_u85_INT(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = EthosU85PipelineINT[input_t](
        ZerosAdd(*init_data),
        input_data(),
        ZerosAdd.aten_op,
        use_to_edge_transform_and_lower=True,
    )
    # Pop the quantization check stage if it exists as no
    # quantization nodes will be present for int + fp inputs.
    if pipeline.has_stage("check.quant_nodes"):
        pipeline.pop_stage("check.quant_nodes")
    pipeline.run()


@common.parametrize(
    "test_data",
    ZerosAdd.test_data_mixed_dtypes,
)
def test_zeros_tosa_INT_not_delegated(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = OpNotSupportedPipeline[input_t](
        ZerosAdd(*init_data),
        input_data(),
        non_delegated_ops={"executorch_exir_dialects_edge__ops_aten_add_Tensor": 1},
        n_expected_delegates=1,
        quantize=True,
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    ZerosAdd.test_data,
)
@common.SkipIfNoModelConverter
def test_zeros_vgf_FP(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = VgfPipeline[input_t](
        ZerosAdd(*init_data), input_data(), ZerosAdd.aten_op, tosa_version="TOSA-1.0+FP"
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    ZerosAdd.test_data,
)
@common.SkipIfNoModelConverter
def test_zeros_vgf_INT(test_data: test_data_t):
    input_data, init_data = test_data
    pipeline = VgfPipeline[input_t](
        ZerosAdd(*init_data),
        input_data(),
        ZerosAdd.aten_op,
        tosa_version="TOSA-1.0+INT",
    )
    # Pop the quantization check stage if it exists as no
    # quantization nodes will be present for int + fp inputs.
    if pipeline.has_stage("check.quant_nodes"):
        pipeline.pop_stage("check.quant_nodes")
    pipeline.run()
